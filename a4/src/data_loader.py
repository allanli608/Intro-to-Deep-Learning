import torch
from torchtext.legacy import data, datasets
from torchtext.vocab import Vectors
import os


class SSTDataPipeline:
    def __init__(self, vector_path=None, batch_size=64, device=None):
        """
        Args:
            vector_path (str, optional): Path to pre-trained vectors (e.g., 'vector.txt').
                                         If None, random embeddings are used.
            batch_size (int): Batch size for iterators.
            device (torch.device): Device to place tensors on.
        """
        self.batch_size = batch_size
        self.device = device if device else torch.device("cpu")
        self.vector_path = vector_path

        # Define Fields
        # include_lengths is needed for packed padded sequences
        self.TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        self.LABEL = data.Field(
            sequential=False, dtype=torch.long, preprocessing=lambda x: int(x) - 1
        )

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.vocab_size = 0
        self.output_dim = 0

    def run(self):
        """Executes the full loading pipeline."""
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

        if self.vector_path:
            # Logic for Pre-trained Embeddings
            print(f"    Loading specific vectors: {self.vector_path}")
            try:
                # Check paths (local or ./data/)
                if os.path.exists(self.vector_path) or os.path.exists(
                    f"./data/{self.vector_path}"
                ):
                    vectors = Vectors(name=self.vector_path, cache="./data")
                    self.TEXT.build_vocab(self.train_data, vectors=vectors)
                    print(f"    Success: Vocab built with '{self.vector_path}'")
                else:
                    print(f"    Error: '{self.vector_path}' not found.")
                    print("    Falling back to random initialization.")
                    self.TEXT.build_vocab(self.train_data)
            except Exception as e:
                print(f"    Error loading vectors: {e}")
                print("    Falling back to random initialization.")
                self.TEXT.build_vocab(self.train_data)
        else:
            # Logic for Random Embeddings (Default)
            print("    No vector_path provided. Initializing random embeddings.")
            self.TEXT.build_vocab(self.train_data)

        # Build label vocab
        self.LABEL.build_vocab(self.train_data)

        self.vocab_size = len(self.TEXT.vocab)
        self.output_dim = len(self.LABEL.vocab)
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

    def get_embeddings(self):
        if self.TEXT.vocab.vectors is not None:
            return self.TEXT.vocab.vectors
        return None

    def get_pad_idx(self):
        return self.TEXT.vocab.stoi[self.TEXT.pad_token]
