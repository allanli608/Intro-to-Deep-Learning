import torch
import torch.nn as nn


class DynamicRNN(nn.Module):
    """
    A Configurable RNN/LSTM/GRU Module.
    Can be used for:
    - Vanilla RNN
    - LSTM
    - GRU
    - Bidirectional
    - Multi-layer
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers=1,
        bidirectional=False,
        dropout=0.5,
        pad_idx=0,
        rnn_type="rnn",
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        rnn_args = {
            "input_size": embedding_dim,
            "hidden_size": hidden_dim,
            "num_layers": n_layers,
            "bidirectional": bidirectional,
            "dropout": dropout if n_layers > 1 else 0,
            "batch_first": True,
        }

        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(**rnn_args)
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(**rnn_args)
        else:
            self.rnn = nn.RNN(**rnn_args)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True
        )

        # Run RNN
        # output contains all hidden states
        # hidden contains the final hidden state (for LSTM it returns hidden, cell)
        if isinstance(self.rnn, nn.LSTM):
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)

        # Handle Bidirectional concatenation
        if self.rnn.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            )
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return self.fc(hidden)
