# Examples

Training examples that validate seqpacker integrations end-to-end.
These require dependencies not managed by the main project.

## Setup

```bash
# Create a separate venv for examples
python -m venv examples/.venv
source examples/.venv/bin/activate

# Install seqpacker (editable from repo root)
pip install -e ..

# Install example dependencies
pip install datasets torch transformers accelerate trl
```

## Run

```bash
# SFTTrainer with pack_dataset (requires GPU)
python examples/sft_trainer.py

# Custom PyTorch training loop (requires GPU)
python examples/pytorch_training.py
```
