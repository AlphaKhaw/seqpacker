# ruff: noqa
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportArgumentType=false, reportCallIssue=false
"""
Packed fine-tuning with TRL SFTTrainer.

Packs the Alpaca dataset with seqpacker and fine-tunes Qwen3-1.7B
using TRL's SFTTrainer. Validates that pack_dataset produces a
correctly formatted Dataset (input_ids, attention_mask, labels,
position_ids) that SFTTrainer consumes without modification.

Dependencies:
    pip install seqpacker datasets torch transformers accelerate trl

Usage:
    python examples/sft_trainer.py
"""

from datasets import load_dataset
from seqpacker.hf_utils import pack_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_SEQ_LEN = 2048


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a real dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")
    texts = [_format_alpaca_sample(ex) for ex in dataset]
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN)

    # One call -- packs sequences and builds a Dataset with correct
    # input_ids, attention_mask, labels, and position_ids
    ds = pack_dataset(
        token_ids=tokenized["input_ids"],
        capacity=MAX_SEQ_LEN,
        padding_value=tokenizer.pad_token_id,
    )

    print(f"Packed {len(texts)} sequences into {len(ds)} bins")
    print(f"Columns: {ds.column_names}")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="/tmp/seqpacker_sft_example",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            dataset_text_field=None,
            remove_unused_columns=False,
        ),
        train_dataset=ds,
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"Final loss: {result.training_loss:.4f}")
    print(f"Steps: {result.global_step}")


if __name__ == "__main__":
    main()
