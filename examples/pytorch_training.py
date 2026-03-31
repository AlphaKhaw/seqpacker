# ruff: noqa
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportArgumentType=false, reportCallIssue=false
"""
Packed training with a custom PyTorch training loop.

Packs the Alpaca dataset with seqpacker and runs a minimal
training loop using packed_collate_fn with a standard DataLoader.
Validates that PackedBatch tensors (input_ids, attention_mask,
labels, position_ids) are shaped correctly and produce valid
gradients.

Dependencies:
    pip install seqpacker datasets torch transformers

Usage:
    python examples/pytorch_training.py
"""

import torch
from datasets import load_dataset
from seqpacker.torch_utils import packed_collate_fn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 32
NUM_STEPS = 5
LR = 1e-4


def _format_alpaca_sample(example):
    """
    Format an Alpaca dataset sample into a single training string.

    Args:
        example (dict): Row from the Alpaca dataset with keys
                        "instruction", "input", and "output".

    Returns:
        str: Formatted prompt-response string.
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a real dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")
    texts = [_format_alpaca_sample(ex) for ex in dataset]
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN)
    token_ids = tokenized["input_ids"]

    # DataLoader with packed collate function
    collate = packed_collate_fn(
        capacity=MAX_SEQ_LEN,
        strategy="obfd",
        padding_value=tokenizer.pad_token_id,
    )
    loader = DataLoader(token_ids, collate_fn=collate, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for step, batch in enumerate(loader):
        if step >= NUM_STEPS:
            break

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        position_ids = batch.position_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        active_labels = (labels != -100).sum().item()
        print(
            f"Step {step + 1}/{NUM_STEPS} | "
            f"loss={loss.item():.4f} | "
            f"batch_shape={input_ids.shape} | "
            f"active_labels={active_labels}"
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
