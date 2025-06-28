import gc
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from .model import Vega7Model

# default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationLoss(nn.Module):
    """Combined loss for distillation and language modeling"""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor
    ):
        student_vocab_size = student_logits.size(-1)
        teacher_vocab_size = teacher_logits.size(-1)
        if student_vocab_size != teacher_vocab_size:
            min_vocab = min(student_vocab_size, teacher_vocab_size)
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]
            labels = labels.clamp(max=min_vocab - 1)

        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (
            self.temperature**2
        )
        student_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)), labels.view(-1)
        )
        return (
            self.alpha * distill_loss + (1 - self.alpha) * student_loss,
            distill_loss,
            student_loss,
        )


def load_teacher_model(model_name: str = "Qwen/Qwen2.5-0.5B"):
    print(f"Loading teacher model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"8-bit loading failed: {e}\nFalling back to full precision.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    return model, tokenizer


def get_config() -> Dict:
    return {
        "state_heads": 4,
        "hidden_size": 256 * 4,
        "n_layers": 8,
        "ffn_size": 2048,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "temperature": 3.0,
        "alpha": 0.7,
        "max_length": 256,
        "gradient_accumulation_steps": 8,
        "num_epochs": 3,
        "warmup_steps": 100,
        "save_steps": 500,
        "eval_steps": 100,
        "max_grad_norm": 1.0,
    }


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}


def get_sample_texts() -> List[str]:
    return [
        "The ARWKV model combines RNN and attention mechanisms for efficient language modeling.",
        "Knowledge distillation transfers knowledge from large models to smaller ones.",
        "RWKV-7 architecture demonstrates strong state tracking capabilities.",
        "This implementation uses the Qwen model as a teacher for distillation.",
        "The time mixing module in RWKV replaces traditional self-attention.",
        "State space models offer an alternative to transformer architectures.",
        "Efficient language models are crucial for deployment on edge devices.",
        "The channel mixing module processes information across feature dimensions.",
    ] * 50


def prepare_data(tokenizer, config: Dict, texts: Optional[List[str]] = None) -> DataLoader:
    if texts is None:
        texts = get_sample_texts()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(texts, tokenizer, max_length=config["max_length"])
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: Dict,
    is_best: bool = False,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    filename = "vega7_best.pt" if is_best else f"vega7_checkpoint_step_{step}.pt"
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


@torch.no_grad()
def generate(
    model: Vega7Model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated = input_ids.clone()
    states = None
    for _ in range(max_length):
        logits, states = model(generated[:, -model.config["max_length"] :], states)
        next_token_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            indices_to_remove = (
                next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            )
            next_token_logits[indices_to_remove] = -float("Inf")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = -float("Inf")
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def train_with_distillation(
    model: Vega7Model, teacher_model, tokenizer, dataloader: DataLoader, config: Dict
):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * config["num_epochs"])
    criterion = DistillationLoss(temperature=config["temperature"], alpha=config["alpha"])

    model.train()
    teacher_model.eval()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits
            student_logits, _ = model(input_ids)
            loss, distill_loss, student_loss = criterion(student_logits, teacher_logits, labels)
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()
            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            epoch_loss += loss.item() * config["gradient_accumulation_steps"]
            progress.set_postfix(
                {"loss": f"{loss.item() * config['gradient_accumulation_steps']:.4f}"}
            )
            if global_step > 0 and global_step % config["save_steps"] == 0:
                save_checkpoint(model, optimizer, epoch, global_step, config)
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch, global_step, config, is_best=True)


def main():
    config = get_config()
    teacher_model, tokenizer = load_teacher_model()
    config["vocab_size"] = teacher_model.config.vocab_size
    model = Vega7Model(config)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    dataloader = prepare_data(tokenizer, config)
    train_with_distillation(model, teacher_model, tokenizer, dataloader, config)
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer_name": teacher_model.config._name_or_path,
    }
    torch.save(final_checkpoint, "vega7_final.pt")
    print("Training complete! Model saved as 'vega7_final.pt'")
    return model, tokenizer


def run_training():
    try:
        return main()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return None, None
