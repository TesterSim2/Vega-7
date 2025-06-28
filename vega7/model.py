import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.layers.append(
                nn.ModuleDict(
                    {
                        "time_mixing": RWKV7TimeMixing(config["hidden_size"], config["n_layers"], i),
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

    def forward(self, input_ids: torch.Tensor, states: list[torch.Tensor] | None = None):
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
