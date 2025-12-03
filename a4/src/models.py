import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Computes a weighted average of all LSTM hidden states.
    This allows the model to 'focus' on specific words (like 'terrible' or 'excellent')
    regardless of where they appear in the sentence.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch size, sent len, hid dim]

        # 1. Calculate Energy
        # energy: [batch size, sent len, 1]
        energy = self.projection(encoder_outputs)

        # 2. Calculate Weights (Softmax)
        # weights: [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1).unsqueeze(-1)

        # 3. Apply Weights to Inputs
        # (batch, 1, sent_len) x (batch, sent_len, hid_dim) -> (batch, 1, hid_dim)
        outputs = (encoder_outputs * weights).sum(dim=1)

        return outputs, weights


class DynamicRNN(nn.Module):
    """
    Updated Configurable Module with Attention and Fine-Tuning controls.
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
        rnn_type="lstm",
        use_attention=False,
        freeze_embeddings=False,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Fine-Tuning Switch: If True, gradients won't be calculated for embeddings
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # 2. RNN Layer
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

        # Determine size of RNN output features
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 3. Attention (Optional)
        if use_attention:
            self.attention = SelfAttention(rnn_output_dim)

        # 4. Output Layer
        self.fc = nn.Linear(rnn_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text: [batch size, sent len]

        embedded = self.dropout(self.embedding(text))

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True
        )

        # Run RNN
        if self.rnn_type == "lstm":
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)

        # Unpack sequence (required for Attention)
        # output: [batch size, sent len, hid dim * num directions]
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # --- Branch: Attention vs. Standard ---
        if self.use_attention:
            # Use the weighted sum of all hidden states
            final_representation, _ = self.attention(output)
        else:
            # Use the final hidden state
            # We have to handle the bidirectional stacking/concatenation manually here
            if self.bidirectional:
                # Concat the final forward and backward hidden layers
                if self.rnn_type == "lstm":
                    # Hidden shape: [num_layers * 2, batch, hid]
                    # We take the last two layers (forward + backward)
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
