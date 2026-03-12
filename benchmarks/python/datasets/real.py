"""
HuggingFace dataset loaders with tokenization and caching.
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from benchmarks.python.metrics.models import DatasetInfo
from benchmarks.python.utils.logging import logger

CACHE_DIR = Path("benchmarks/cache")


class RealDataLoader:
    """
    Loader for real-world datasets from HuggingFace Hub.

    Tokenizes text to extract sequence lengths and caches results
    to benchmarks/cache/ for fast reuse across benchmark runs.
    """

    DATASET_CONFIGS: dict = {
        "alpaca": {
            "path": "yahma/alpaca-cleaned",
            "config": None,
            "split": "train",
            "text_columns": ["instruction", "input", "output"],
            "streaming": False,
        },
        "ultrachat": {
            "path": "HuggingFaceH4/ultrachat_200k",
            "config": None,
            "split": "train_sft",
            "text_columns": ["messages"],
            "streaming": False,
            "is_chat": True,
        },
        "c4": {
            "path": "allenai/c4",
            "config": "en",
            "split": "train",
            "text_columns": ["text"],
            "streaming": True,
        },
    }

    @staticmethod
    def _cache_key(
        dataset_name: str,
        tokenizer_name: str,
        max_samples: int,
    ) -> str:
        """
        Generate a cache key for tokenized lengths.

        Args:
            dataset_name (str): Name of the dataset.
            tokenizer_name (str): Name of the tokenizer.
            max_samples (int): Maximum number of samples.

        Returns:
            str: Cache file name.
        """
        raw = f"{dataset_name}_{tokenizer_name}_{max_samples}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{dataset_name}_{h}.json"

    @staticmethod
    def _extract_text(example: dict, config: dict) -> str:
        """
        Extract text from a dataset example based on config.

        Args:
            example (dict): A single dataset example.
            config (dict): Dataset configuration.

        Returns:
            str: Extracted text string.
        """
        if config.get("is_chat"):
            messages = example.get("messages", [])
            return " ".join(
                msg.get("content", "") for msg in messages if isinstance(msg, dict)
            )
        columns = config["text_columns"]
        parts = [str(example.get(col, "")) for col in columns]
        return " ".join(part for part in parts if part)

    @classmethod
    def available_datasets(cls) -> list[str]:
        """
        Return the list of available dataset names.

        Returns:
            list[str]: Available dataset names.
        """
        return list(cls.DATASET_CONFIGS.keys())

    @classmethod
    def load(
        cls,
        dataset_name: str,
        max_samples: int = 10000,
        tokenizer_name: str = "gpt2",
    ) -> DatasetInfo:
        """
        Load a real dataset from HuggingFace and tokenize to get sequence lengths.

        Caches tokenized lengths to benchmarks/cache/ for fast reuse.

        Args:
            dataset_name (str): Key into DATASET_CONFIGS (e.g., 'alpaca', 'c4').
            max_samples (int): Maximum number of samples to load.
            tokenizer_name (str): HuggingFace tokenizer name.

        Returns:
            DatasetInfo: Dataset with tokenized sequence lengths.

        Raises:
            ValueError: If dataset_name is not in DATASET_CONFIGS.
        """
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {cls.available_datasets()}"
            )

        config = cls.DATASET_CONFIGS[dataset_name]

        # Check cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / cls._cache_key(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
        )
        if cache_file.exists():
            logger.info(f"Loading cached lengths from {cache_file}")
            cached = json.loads(cache_file.read_text())
            return DatasetInfo(name=dataset_name, lengths=cached["lengths"])

        logger.info(f"Loading dataset '{dataset_name}' from HuggingFace...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load dataset
        load_kwargs: dict = {"path": config["path"], "split": config["split"]}
        if config["config"]:
            load_kwargs["name"] = config["config"]
        if config["streaming"]:
            load_kwargs["streaming"] = True

        ds = load_dataset(**load_kwargs)

        # Extract examples
        if config["streaming"]:
            examples = []
            for i, example in enumerate(ds):
                if i >= max_samples:
                    break
                examples.append(example)
        else:
            if len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            examples = list(ds)

        logger.info(f"Tokenizing {len(examples)} examples with '{tokenizer_name}'...")

        # Tokenize examples
        lengths: list[int] = []
        for example in examples:
            text = cls._extract_text(example=example, config=config)
            if text.strip():
                tokens = tokenizer.encode(text)
                lengths.append(len(tokens))

        # Cache results
        cache_file.write_text(json.dumps({"lengths": lengths}))
        logger.info(f"Cached {len(lengths)} lengths to {cache_file}")

        return DatasetInfo(name=dataset_name, lengths=lengths)
