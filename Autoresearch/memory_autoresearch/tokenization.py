from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer

from .cache import tokenizer_root
from .config import DEFAULT_MAX_SEQUENCE_LENGTH


@dataclass
class TokenizedBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray


class BertTokenizerAdapter:
    def __init__(self, checkpoint: str, max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH):
        self.checkpoint = checkpoint
        self.max_sequence_length = max_sequence_length
        cache_dir = tokenizer_root() / checkpoint.replace("/", "__")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def encode_texts(self, texts: list[str]) -> TokenizedBatch:
        encoded = self.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(encoded["input_ids"], dtype=np.int32)
        return TokenizedBatch(
            input_ids=encoded["input_ids"].astype(np.int32),
            attention_mask=encoded["attention_mask"].astype(np.int32),
            token_type_ids=token_type_ids.astype(np.int32),
        )

    def encode_pairs(self, queries: list[str], documents: list[str]) -> TokenizedBatch:
        encoded = self.tokenizer(
            queries,
            documents,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
        )
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(encoded["input_ids"], dtype=np.int32)
        return TokenizedBatch(
            input_ids=encoded["input_ids"].astype(np.int32),
            attention_mask=encoded["attention_mask"].astype(np.int32),
            token_type_ids=token_type_ids.astype(np.int32),
        )
