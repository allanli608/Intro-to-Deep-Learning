import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):

        energy = self.projection(encoder_outputs)

        weights = F.softmax(energy.squeeze(-1), dim=1).unsqueeze(-1)

        outputs = (encoder_outputs * weights).sum(dim=1)

        return outputs, weights


class DynamicRNN(nn.Module):
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
        rnn_type="lstm",
        use_attention=False,
        freeze_embeddings=False,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        rnn_args = {
            "input_size": embedding_dim,
            "hidden_size": hidden_dim,
            "num_layers": n_layers,
            "bidirectional": bidirectional,
            "dropout": dropout if n_layers > 1 else 0,
            "batch_first": True,
        }

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_args)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_args)
        else:
            self.rnn = nn.RNN(**rnn_args)

        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if use_attention:
            self.attention = SelfAttention(rnn_output_dim)

        self.fc = nn.Linear(rnn_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True
        )

        if self.rnn_type == "lstm":
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        if self.use_attention:
            final_representation, _ = self.attention(output)
        else:
            if self.bidirectional:
                if self.rnn_type == "lstm":
                    final_representation = torch.cat(
                        (hidden[-2, :, :], hidden[-1, :, :]), dim=1
                    )
                else:
                    final_representation = torch.cat(
                        (hidden[-2, :, :], hidden[-1, :, :]), dim=1
                    )
            else:
                final_representation = hidden[-1, :, :]

        return self.fc(self.dropout(final_representation))
