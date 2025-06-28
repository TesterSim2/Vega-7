import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankStateMixing(nn.Module):
    """Matrix-valued state with active and ghost blocks."""

    def __init__(
        self,
        hidden_size: int,
        heads: int = 4,
        rank: int = 1,
        ghost_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.rank = rank
        self.ghost_dtype = ghost_dtype

        # Parameter generator for rank-1 update
        self.param_generator = nn.Linear(hidden_size, heads + self.head_dim)
        self.ghost_update_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ghost_update_transform = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_x = nn.LayerNorm(hidden_size)

    def init_state(self, batch_size: int, device: torch.device):
        active = torch.zeros(batch_size, self.heads, self.head_dim, device=device)
        ghost = torch.zeros(
            batch_size,
            self.heads,
            self.head_dim,
            device=device,
            dtype=self.ghost_dtype,
        )
        return active, ghost

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        B, T, _ = x.size()
        x = self.ln_x(x)
        r = self.receptance(x)
        if state is None:
            active, ghost = self.init_state(B, x.device)
        else:
            active, ghost = state

        outputs = []
        for t in range(T):
            xt = x[:, t]
            params = self.param_generator(xt)
            u, v = params.split([self.heads, self.head_dim], dim=-1)
            active = active + u.unsqueeze(-1) * v.unsqueeze(1)

            gate = torch.sigmoid(self.ghost_update_gate(xt)).view(B, self.heads, self.head_dim)
            transform = torch.tanh(self.ghost_update_transform(xt)).view(
                B, self.heads, self.head_dim
            )
            ghost = ghost + (gate * transform).to(self.ghost_dtype)

            combined = active + ghost.to(dtype=active.dtype)
            rt = r[:, t].view(B, self.heads, self.head_dim).sigmoid()
            out = torch.einsum("bhd,bhd->bd", rt, combined)
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        output = self.output(output)
        return output, (active, ghost)


class RWKV7TimeMixing(nn.Module):
    """Time mixing module replacing self-attention"""

    def __init__(self, hidden_size: int, n_layer: int, layer_id: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.layer_id = layer_id

        self.time_w = nn.Parameter(torch.ones(hidden_size))
        self.time_decay = nn.Parameter(torch.zeros(hidden_size))
        self.time_first = nn.Parameter(torch.zeros(hidden_size))

        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

        self.ln_x = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        B, T, C = x.size()
        x = self.ln_x(x)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        if state is None:
            state = torch.zeros(B, C, C, device=x.device)

        w = torch.exp(-torch.exp(self.time_decay))
        outputs = []
        for t in range(T):
            kt = k[:, t]
            vt = v[:, t]
            rt = r[:, t]
            state = state * w.unsqueeze(0).unsqueeze(-1) + kt.unsqueeze(-1) * vt.unsqueeze(1)
            out = torch.einsum("bc,bcd->bd", rt.sigmoid(), state)
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        output = self.output(output)
        return output, state


class ChannelMixing(nn.Module):
    """Channel mixing module"""

    def __init__(self, hidden_size: int, layer_id: int, ffn_size: int | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        ffn_size = ffn_size or hidden_size * 4

        self.key = nn.Linear(hidden_size, ffn_size, bias=False)
        self.value = nn.Linear(ffn_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_x = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        x = self.ln_x(x)
        k = self.key(x)
        k = F.relu(k) ** 2
        kv = self.value(k)
        return x * self.receptance(x).sigmoid() + kv


class Vega7Model(nn.Module):
    """Vega-7 model with RWKV7 architecture"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])

        self.layers = nn.ModuleList()
        for i in range(config["n_layers"]):
            if "state_heads" in config:
                time_mixing = LowRankStateMixing(
                    config["hidden_size"],
                    heads=config["state_heads"],
                    rank=1,
                )
            else:
                time_mixing = RWKV7TimeMixing(
                    config["hidden_size"], config["n_layers"], i
                )
            self.layers.append(
                nn.ModuleDict(
                    {
                        "time_mixing": time_mixing,
                        "channel_mixing": ChannelMixing(
                            config["hidden_size"],
                            i,
                            config.get("ffn_size", config["hidden_size"] * 4),
                        ),
                    }
                )
            )

        self.ln_out = nn.LayerNorm(config["hidden_size"])
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(
        self, input_ids: torch.Tensor, states: list | None = None
    ):
        x = self.embeddings(input_ids)
        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        for i, layer in enumerate(self.layers):
            time_out, new_state = layer["time_mixing"](x, states[i])
            x = x + time_out
            new_states.append(new_state)
            x = x + layer["channel_mixing"](x)

        x = self.ln_out(x)
        logits = self.head(x)
        return logits, new_states
