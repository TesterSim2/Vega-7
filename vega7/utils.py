import gc
import torch
from .model import Vega7Model


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = Vega7Model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model with config: {config}")
    return model, config
