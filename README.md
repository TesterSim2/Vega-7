# Vega-7

This repository contains a prototype implementation of the **Vega‑7** model described in the paper *"ARWKV: Pretrain is not what we need, an RNN‑Attention‑Based Language Model Born from Transformer"*.

All training utilities are provided as regular Python modules rather than notebooks.

## Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The training script loads the teacher model with 8-bit weights using
`device_map="auto"`, so the [Hugging Face Accelerate](https://github.com/huggingface/accelerate)
library is required.

Run a small distillation demo:

```bash
python -m vega7
```

The script downloads a small Qwen model from HuggingFace and distills it into a lightweight RWKV‑style network. Configuration and helper functions live in `vega7/`.
