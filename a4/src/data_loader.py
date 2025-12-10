"""
Utilities for working with the Stanford Sentiment Treebank (SST) dataset without
depending on torchtext. The script keeps the original HW4 workflow: it builds a
vocabulary with pretrained embeddings, inspects the dataset, and creates
iterators that yield padded batches ready for training in PyTorch.
"""

from __future__ import annotations

import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
FINE_GRAINED_LABELS = [
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
]
COARSE_LABELS = ["negative", "neutral", "positive"]


@dataclass
class SSTExample:
    text: List[str]
    label: str


class Vocabulary:
    """Basic vocabulary that mirrors the small subset of torchtext features we need."""

    def __init__(
        self,
        pad_token: Optional[str] = PAD_TOKEN,
        unk_token: Optional[str] = UNK_TOKEN,
        min_freq: int = 1,
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq
        self.specials = [tok for tok in (unk_token, pad_token) if tok is not None]
        self.freqs: Counter = Counter()
        self.itos: List[str] = []
        self.stoi: dict[str, int] = {}
        self.vectors: torch.Tensor = torch.empty(0)

    def build(
        self, iterator: Iterable[Sequence[str]], vectors_path: Optional[str] = None
    ) -> None:
        self.freqs = Counter()
        for tokens in iterator:
            self.freqs.update(tokens)

        self.itos = []
        self.stoi = {}

        for token in self.specials:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        for token, freq in self.freqs.most_common():
            if freq < self.min_freq or token in self.stoi:
                continue
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

        if vectors_path:
            self.vectors = self._build_vectors(vectors_path)
        else:
            self.vectors = torch.empty(len(self.itos), 0)

    def __len__(self) -> int:
        return len(self.itos)

    def _build_vectors(self, path: str) -> torch.Tensor:
        embeddings: dict[str, torch.Tensor] = {}
        dim: Optional[int] = None

        with open(path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                word, values = parts[0], parts[1:]
                vector = torch.tensor([float(v) for v in values], dtype=torch.float)
                if dim is None:
                    dim = vector.size(0)
                embeddings[word] = vector

        if dim is None:
            raise ValueError(f"No vectors found in {path}")

        vectors = torch.empty(len(self.itos), dim)
        torch.nn.init.uniform_(vectors, -0.05, 0.05)

        for idx, token in enumerate(self.itos):
            if token in embeddings:
                vectors[idx] = embeddings[token]

        return vectors


class SimpleField:
    """A very small stand-in for torchtext.data.Field."""

    def __init__(
        self,
        sequential: bool = True,
        dtype: torch.dtype = torch.long,
        pad_token: Optional[str] = PAD_TOKEN,
        unk_token: Optional[str] = UNK_TOKEN,
    ) -> None:
        self.sequential = sequential
        self.dtype = dtype
        pad_token = pad_token if sequential else None
        self.vocab = Vocabulary(pad_token=pad_token, unk_token=unk_token)

    def build_vocab(
        self, dataset: "SSTDataset", vectors_path: Optional[str] = None
    ) -> None:
        if self.sequential:
            iterator = (example.text for example in dataset.examples)
            self.vocab.build(iterator, vectors_path=vectors_path)
        else:
            iterator = ([example.label] for example in dataset.examples)
            self.vocab.build(iterator, vectors_path=None)


class SSTDataset(Dataset):
    def __init__(
        self, examples: List[SSTExample], fields: dict[str, SimpleField]
    ) -> None:
        self.examples = examples
        self.fields = fields

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SSTExample:
        return self.examples[idx]


def ensure_sst_tree_dir(root: str) -> Path:
    root_path = Path(root)
    tree_dir = root_path / "sst" / "trees"
    if tree_dir.exists():
        return tree_dir

    zip_path = root_path / "sst" / "trainDevTestTrees_PTB.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root_path / "sst")
        return tree_dir

    raise FileNotFoundError(
        f"Could not find SST data in {tree_dir}. "
        "Please place trainDevTestTrees_PTB.zip under .data/sst/ or extract it manually."
    )


def parse_tree_examples(
    tree: str, include_subtrees: bool
) -> List[Tuple[int, List[str]]]:
    tokens = tree.replace("(", " ( ").replace(")", " ) ").split()
    index = 0
    examples: List[Tuple[int, List[str]]] = []

    def parse_node() -> Tuple[int, List[str]]:
        nonlocal index
        assert tokens[index] == "("
        index += 1
        label = int(tokens[index])
        index += 1
        words: List[str] = []

        while index < len(tokens) and tokens[index] != ")":
            if tokens[index] == "(":
                child_label, child_words = parse_node()
                words.extend(child_words)
                if include_subtrees and child_words:
                    examples.append((child_label, child_words.copy()))
            else:
                words.append(tokens[index])
                index += 1

        index += 1
        return label, words

    label, words = parse_node()
    examples.append((label, words))
    return examples


def map_label(label_id: int, fine_grained: bool) -> str:
    if fine_grained:
        return FINE_GRAINED_LABELS[label_id]
    if label_id <= 1:
        return COARSE_LABELS[0]
    if label_id == 2:
        return COARSE_LABELS[1]
    return COARSE_LABELS[2]


def read_sst_file(
    path: Path, fine_grained: bool = True, include_subtrees: bool = False
) -> List[SSTExample]:
    examples: List[SSTExample] = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for label_id, tokens in parse_tree_examples(line, include_subtrees):
                if not tokens:
                    continue
                label = map_label(label_id, fine_grained)
                examples.append(SSTExample(tokens, label))
    return examples


def load_sst_splits(
    text_field: SimpleField,
    label_field: SimpleField,
    root: str = ".data",
    train_subtrees: bool = False,
    fine_grained: bool = True,
    train: str = "train.txt",
    validation: str = "dev.txt",
    test: str = "test.txt",
) -> Tuple[SSTDataset, SSTDataset, SSTDataset]:
    tree_dir = ensure_sst_tree_dir(root)
    fields = {"text": text_field, "label": label_field}

    train_examples = read_sst_file(
        tree_dir / train, fine_grained, include_subtrees=train_subtrees
    )
    val_examples = read_sst_file(
        tree_dir / validation, fine_grained, include_subtrees=False
    )
    test_examples = read_sst_file(tree_dir / test, fine_grained, include_subtrees=False)

    return (
        SSTDataset(train_examples, fields),
        SSTDataset(val_examples, fields),
        SSTDataset(test_examples, fields),
    )


@dataclass
class Batch:
    text: torch.Tensor
    lengths: torch.Tensor
    label: torch.Tensor


def resolve_device(device: Optional[int | str | torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        if device < 0 or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cpu")


def make_collate_fn(
    text_field: SimpleField, label_field: SimpleField, device: torch.device
) -> Callable[[Sequence[SSTExample]], Batch]:
    pad_idx = text_field.vocab.stoi.get(PAD_TOKEN, 0)
    unk_idx = text_field.vocab.stoi.get(UNK_TOKEN, pad_idx)

    def numericalize(tokens: Sequence[str]) -> torch.Tensor:
        indices = [text_field.vocab.stoi.get(token, unk_idx) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def collate(batch: Sequence[SSTExample]) -> Batch:

        batch = sorted(batch, key=lambda x: len(x.text), reverse=True)
        text_tensors = [numericalize(example.text) for example in batch]

        lengths = torch.tensor(
            [tensor.size(0) for tensor in text_tensors], dtype=torch.long
        )

        padded = pad_sequence(text_tensors, batch_first=True, padding_value=pad_idx)
        labels = torch.tensor(
            [label_field.vocab.stoi.get(example.label, 0) for example in batch],
            dtype=label_field.dtype,
        )

        return Batch(
            text=padded.to(device),
            lengths=lengths,
            label=labels.to(device),
        )

    return collate


def build_iterators(
    datasets: Sequence[SSTDataset],
    text_field: SimpleField,
    label_field: SimpleField,
    batch_size: int = 64,
    device: Optional[int | str | torch.device] = None,
) -> Tuple[DataLoader, ...]:
    resolved_device = resolve_device(device)
    collate_fn = make_collate_fn(text_field, label_field, resolved_device)

    loaders: List[DataLoader] = []
    for idx, split in enumerate(datasets):
        shuffle = idx == 0
        loaders.append(
            DataLoader(
                split,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )
        )
    return tuple(loaders)


def get_sst_data_loaders(batch_size=64, vector_path="./data/vector.txt", device="cpu"):
    print("--> Initializing Fields...")
    TEXT = SimpleField()
    LABEL = SimpleField(
        sequential=False, dtype=torch.long, pad_token=None, unk_token=UNK_TOKEN
    )

    print("--> Loading SST Splits...")
    train, val, test = load_sst_splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False
    )

    print(f"    Training samples: {len(train)}")

    print("--> Building Vocabulary...")
    TEXT.build_vocab(train, vectors_path=vector_path)
    LABEL.build_vocab(train)
    LABEL.vocab.stoi = {
        "very negative": 1,
        "negative": 2,
        "neutral": 3,
        "positive": 4,
        "very positive": 5,
    }

    print(f"    Vocab Size: {len(TEXT.vocab)}")
    print(f"    Label Map: {LABEL.vocab.stoi}")

    print("--> Building Iterators...")
    train_iter, val_iter, test_iter = build_iterators(
        (train, val, test), TEXT, LABEL, batch_size=batch_size, device=device
    )

    return train_iter, val_iter, test_iter, TEXT, LABEL
