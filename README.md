# Vega-7

This repository contains a prototype implementation of the "Vega 7" model described in the paper **ARWKV: Pretrain is not what we need, an RNN-Attention-Based Language Model Born from Transformer**. The code is provided as a demonstration and loads the Qwen teacher model to distill it into a smaller RNN-based architecture.

## Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run the training demo:

```bash
python -m vega7
```

This will attempt to download the Qwen model from HuggingFace and run a short distillation and fine-tuning loop. Adjust the constants in `vega7/__init__.py` for different settings.
