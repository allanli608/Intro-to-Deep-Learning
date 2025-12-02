import torch
from torchtext.legacy import data, datasets
from torchtext.vocab import Vectors
import os


class SSTDataPipeline:
    def __init__(self, vector_path=None, batch_size=64, device=None):
        self.batch_size = batch_size
        self.device = device if device else torch.device("cpu")
        self.vector_path = vector_path

        self.TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)

        # FIX: Remove 'preprocessing' and 'use_vocab=False'
        # We let it build a vocab, but we will control the order.
        self.LABEL = data.Field(sequential=False, dtype=torch.long)

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.vocab_size = 0
        self.output_dim = 0

    def run(self):
        self._load_data()
        self._build_vocab()
        return self._get_iterators()

    def _load_data(self):
        print("--> Loading SST splits...")
        self.train_data, self.val_data, self.test_data = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=True, train_subtrees=False
        )

    def _build_vocab(self):
        print("--> Building Vocabulary...")

        # 1. Build Text Vocab (Same as before)
        if self.vector_path:
            print(f"    Loading specific vectors: {self.vector_path}")
            try:
                if os.path.exists(self.vector_path) or os.path.exists(
                    f"./data/{self.vector_path}"
                ):
                    vectors = Vectors(name=self.vector_path, cache="./data")
                    self.TEXT.build_vocab(self.train_data, vectors=vectors)
                    print(f"    Success: Vocab built with '{self.vector_path}'")
                else:
                    print(f"    Error: '{self.vector_path}' not found. Using random.")
                    self.TEXT.build_vocab(self.train_data)
            except Exception as e:
                print(f"    Error loading vectors: {e}. Using random.")
                self.TEXT.build_vocab(self.train_data)
        else:
            print("    No vector_path provided. Initializing random embeddings.")
            self.TEXT.build_vocab(self.train_data)

        # 2. FIX: Manually build Label Vocab to ensure 0=Very Negative, 4=Very Positive
        # Note: TorchText often pads, but since sequential=False, it shouldn't produce a <pad> token
        # unless specified. However, we simply build it from the data to be safe.
        self.LABEL.build_vocab(self.train_data)

        # CRITICAL: Overwrite the stoi (string-to-index) dictionary to force your order
        # This maps the text label to the integer index you want.
        custom_stoi = {
            "very negative": 0,
            "negative": 1,
            "neutral": 2,
            "positive": 3,
            "very positive": 4,
        }
        self.LABEL.vocab.stoi = custom_stoi
        # We also need to update itos (index-to-string) to match
        self.LABEL.vocab.itos = [
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ]

        self.vocab_size = len(self.TEXT.vocab)
        self.output_dim = len(self.LABEL.vocab)

        # VERIFICATION
        print(f"    Label Mapping: {self.LABEL.vocab.stoi}")
        print(f"    Vocab Size: {self.vocab_size}")

    def _get_iterators(self):
        print("--> Creating Iterators...")
        return data.BucketIterator.splits(
            (self.train_data, self.val_data, self.test_data),
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=self.device,
        )

    # Helper methods remain the same...
    def get_embeddings(self):
        return self.TEXT.vocab.vectors

    def get_pad_idx(self):
        return self.TEXT.vocab.stoi[self.TEXT.pad_token]
