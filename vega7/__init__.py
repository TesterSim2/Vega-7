
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc # Garbage Collector for memory management

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==============================================================================

# --- Teacher Model ---
TEACHER_MODEL_NAME = "Qwen/Qwen2-0.5B" # Using 0.5B for Colab. Change to 4B if you have a high-RAM GPU.

# --- Student Model (Vega 7) ---
# NOTE: hidden_size and vocab_size will be automatically set from the teacher model
STUDENT_NUM_LAYERS = 8          # Number of LatentThinkingBlocks
ACTIVE_CHANNELS = 128           # (GhostRNN) Number of matrix rows for expensive updates. Must be < hidden_size
NUM_THINKING_STEPS = 3          # (Latent Reasoning) Inner loop steps per token.

# --- Training ---
LEARNING_RATE = 1e-4
NUM_DISTILLATION_EPOCHS = 15
NUM_SFT_EPOCHS = 3
TEMPERATURE = 2.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. VEGA 7 STUDENT MODEL DEFINITION (CORRECTED)
# ==============================================================================

class DynamicRecurrenceModule(nn.Module):
    """
    Implements the Finch/Goose + GhostRNN dynamic recurrence.
    It updates a matrix-valued state [H, H] based on an input vector x [H].
    """
    def __init__(self, hidden_size, active_channels):
        super().__init__()
        if active_channels >= hidden_size:
            raise ValueError("active_channels must be smaller than hidden_size")
        self.hidden_size = hidden_size
        self.active_channels = active_channels
        self.ghost_channels = hidden_size - active_channels

        # (Finch/Goose) Network to generate dynamic parameters for a low-rank update
        self.param_generator = nn.Linear(hidden_size, 2 * hidden_size * active_channels // hidden_size)

        # (GhostRNN) Cheap, shared linear transformation for the ghost part of the state
        self.ghost_update_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ghost_update_transform = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, state, x):
        # state shape: [batch, hidden_size, hidden_size]
        # x shape: [batch, hidden_size]

        active_state, ghost_state = torch.split(state, [self.active_channels, self.ghost_channels], dim=1)

        # 1. Expensive, Dynamic Update (Low-Rank Update)
        params = self.param_generator(x)
        rank = params.shape[-1] // (2 * self.hidden_size)
        u, v = torch.chunk(params, 2, dim=-1)
        u = u.view(-1, self.active_channels, rank)
        v = v.view(-1, rank, self.hidden_size)
        active_update = u @ v

        # 2. Cheap, Shared Update
        gate = torch.sigmoid(self.ghost_update_gate(ghost_state))
        transform = torch.tanh(self.ghost_update_transform(ghost_state))
        ghost_update = gate * transform

        new_state = state + torch.cat([active_update, ghost_update], dim=1)
        return new_state

class LatentThinkingBlock(nn.Module):
    """
    A single layer of the Vega 7 model. Corrected to process one token at a time.
    """
    def __init__(self, hidden_size, active_channels, num_thinking_steps):
        super().__init__()
        self.num_thinking_steps = num_thinking_steps

        self.norm_x = nn.LayerNorm(hidden_size)
        self.norm_state = nn.LayerNorm(hidden_size)

        self.recurrence = DynamicRecurrenceModule(hidden_size, active_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, state, x):
        # state shape: [batch, H, H], x shape: [batch, H]

        # Use x to update the state. This is the "Recurrence" part of the block.
        # This is analogous to the attention block in a Transformer.
        state = self.recurrence(state, self.norm_x(x))

        # Latent Thinking Loop
        if self.num_thinking_steps > 1:
            null_input = torch.zeros_like(x)
            for _ in range(self.num_thinking_steps - 1):
                state = self.recurrence(state, null_input)

        # Project state back to a vector and add to the main residual stream
        state_vector = torch.mean(state, dim=1)
        x = x + state_vector

        # Apply feed-forward layer to the result, also with a residual connection
        x = x + self.feed_forward(self.norm_state(x))
        return state, x

class Vega7Model(nn.Module):
    """
    The complete Vega 7 student model with the corrected forward pass.
    """
    def __init__(self, vocab_size, hidden_size, num_layers, active_channels, num_thinking_steps):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LatentThinkingBlock(hidden_size, active_channels, num_thinking_steps)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input_ids, states):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        outputs = []
        # **CORRECTED LOGIC**: Iterate over the sequence dimension (time)
        for t in range(seq_len):
            token_repr = x[:, t, :]
            new_layer_states = []
            for i, layer in enumerate(self.layers):
                # Get the state for the current layer
                layer_state = states[i]
                # Update the state and token representation
                new_state, token_repr = layer(layer_state, token_repr)
                new_layer_states.append(new_state)
            
            # The output of the final layer for this timestep is collected
            outputs.append(token_repr)
            # The list of states is updated for the next timestep
            states = new_layer_states

        # Stack the outputs from each timestep to form the final sequence tensor
        final_output = torch.stack(outputs, dim=1)
        final_output = self.output_norm(final_output)
        logits = self.output_head(final_output)

        return logits, states

# --- The rest of the script (Loss, Setup, Training) is largely the same, ---
# --- but with corrected state handling in the training loops.             ---

# ==============================================================================
# 3. KNOWLEDGE DISTILLATION SETUP
# ==============================================================================

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        return self.kl_div(student_log_probs, teacher_probs)


def main():
    print(f"Using device: {DEVICE}")
    if "cuda" in DEVICE:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    print(f"Loading teacher model: {TEACHER_MODEL_NAME}...")
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True).to(DEVICE)
        teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)
        teacher_model.eval()
        print("Teacher model loaded successfully.")
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        teacher_model = None

    if teacher_model:
        STUDENT_HIDDEN_SIZE = teacher_model.config.hidden_size
        STUDENT_VOCAB_SIZE = teacher_model.config.vocab_size

        student_model = Vega7Model(
            vocab_size=STUDENT_VOCAB_SIZE,
            hidden_size=STUDENT_HIDDEN_SIZE,
            num_layers=STUDENT_NUM_LAYERS,
            active_channels=ACTIVE_CHANNELS,
            num_thinking_steps=NUM_THINKING_STEPS
        ).to(DEVICE)

        print("\n--- Student Model (Vega 7) Initialized ---")
        print(f"  - Hidden Size: {STUDENT_HIDDEN_SIZE}, Vocab Size: {STUDENT_VOCAB_SIZE}")

        optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
        distillation_loss_fn = KnowledgeDistillationLoss(temperature=TEMPERATURE)

        text = "The Qwen model, a large language model developed by Alibaba Cloud, has demonstrated exceptional performance on various benchmarks."
        inputs = teacher_tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]

        # ==============================================================================
        # 4. DISTILLATION TRAINING LOOP (CORRECTED)
        # ==============================================================================
        if student_model:
            print("\n--- Starting Knowledge Distillation ---")

            for epoch in range(NUM_DISTILLATION_EPOCHS):
                student_model.train()

                # **CORRECTED**: Initialize state at the beginning of each epoch
                batch_size = input_ids.shape[0]
                states = [
                    torch.zeros(batch_size, STUDENT_HIDDEN_SIZE, STUDENT_HIDDEN_SIZE).to(DEVICE)
                    for _ in range(STUDENT_NUM_LAYERS)
                ]

                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids)
                    teacher_logits = teacher_outputs.logits

                student_logits, _ = student_model(input_ids, states)
                loss = distillation_loss_fn(student_logits, teacher_logits)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()

                print(f"Epoch {epoch + 1}/{NUM_DISTILLATION_EPOCHS}, Loss: {loss.item():.6f}")

            print("--- Distillation Finished ---")

        # ==============================================================================
        # 5. SUPERVISED FINE-TUNING (SFT) (CORRECTED)
        # ==============================================================================
            def supervised_fine_tuning(model, tokenizer, dataset, num_epochs=3):
                print("\n--- Starting Supervised Fine-Tuning ---")
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE / 5)
                loss_fn = nn.CrossEntropyLoss()

                for epoch in range(num_epochs):
                    for i, batch in enumerate(dataset):
                        input_ids = tokenizer(batch["text"], return_tensors="pt")["input_ids"].to(DEVICE)
                        labels = input_ids.clone()

                        batch_size = input_ids.shape[0]
                        sft_states = [
                            torch.zeros(batch_size, model.hidden_size, model.hidden_size).to(DEVICE)
                            for _ in range(model.num_layers)
                        ]

                        logits, _ = model(input_ids, sft_states)
                        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        print(f"SFT Epoch {epoch + 1}, Batch {i+1}, Loss: {loss.item():.4f}")

            dummy_sft_dataset = [{"text": "Instruction: Write a poem about AI.\nOutput: In circuits of silicon, a mind begins to bloom."}]
            supervised_fine_tuning(student_model, teacher_tokenizer, dummy_sft_dataset, num_epochs=NUM_SFT_EPOCHS)

        # ==============================================================================
        # 6. SAVE THE FINAL MODEL
        # ==============================================================================
            FINAL_MODEL_PATH = "vega7_distilled_model.pt"
            torch.save(student_model.state_dict(), FINAL_MODEL_PATH)
            print("\n--- Model Saved ---")
            print(f"Final Vega 7 model state dictionary saved to: {FINAL_MODEL_PATH}")

            del teacher_model
            del student_model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
