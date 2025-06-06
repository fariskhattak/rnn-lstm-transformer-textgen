import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        nhead=4,
        num_layers=2,
        feedforward_dim=512,
        dropout=0.2,
        pad_token_id=0,
        max_seq_len=512,
    ):
        """
        Create a Transformer-based language model with sinusoidal positional encoding.

        :param vocab_size:       Size of the vocabulary
        :param embed_dim:        Dimensionality for token embeddings
        :param nhead:            Number of attention heads per Transformer layer
        :param num_layers:       Number of Transformer encoder layers
        :param feedforward_dim:  Size of the feedforward network inside each layer
        :param dropout:          Dropout probability throughout the model
        :param pad_token_id:     Padding token ID
        :param max_seq_len:      Maximum sequence length for positional encoding
        """
        super().__init__()

        # Embedding for token IDs
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_token_id
        )

        # PositionalEncoding layer
        self.pos_encoding = PositionalEncoding(
            d_model=embed_dim, dropout=dropout, max_len=max_seq_len
        )

        # Define the Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final projection to vocabulary
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        """
        Forward pass for the Transformer language model.

        :param input_ids: (batch_size, seq_len) tensor of token IDs
        :param attention_mask: Optional mask for ignoring <pad> tokens
        :return:
            logits, hidden
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        # Token Embedding
        x = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)

        # Positional Encoding
        x = self.pos_encoding(x)

        # Causal mask (look-ahead) for self-attention
        attention_mask = self.create_causal_mask(seq_len, device=device)  # (seq_len, seq_len)

        # Pass through Transformer Encoder
        encoded = self.transformer_encoder(x, mask=attention_mask)

        # Project to vocabulary
        logits = self.fc(encoded)  # (batch_size, seq_len, vocab_size)
        return logits, None

    def predict_next_token(self, input_ids, top_p=0.9):
        """
        Generate the next token from the last position in input_ids.
        Transformers do not hold a hidden state, so we re-run forward each time.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            # Take last token's logits
            last_token_logits = logits[:, -1, :]
            probs = F.softmax(last_token_logits, dim=-1)
            # next_token_id = torch.argmax(probs, dim=-1)

            # Apply top-p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create a mask for tokens to keep
            sorted_mask = cumulative_probs < top_p
            sorted_mask[:, 0] = True  # always keep at least one token

            # Zero out probabilities for tokens outside top-p
            filtered_probs = sorted_probs * sorted_mask
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)  # renormalize

            sampled_index = torch.multinomial(filtered_probs, num_samples=1)  # shape: (batch_size, 1)

            # Map sampled indices back to original vocab indices
            next_token_id = sorted_indices.gather(1, sampled_index)  # shape: (batch_size, 1)

            return next_token_id.item(), None
        
    def create_causal_mask(self, seq_len, device="cuda"):
        """
        Creates a causal (look-ahead) mask for self-attention.
        
        Returns:
            (seq_len, seq_len) BoolTensor where True indicates masking.
        """
        # Shape: (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, device="cuda"):
        """
        Autoregressive text generation using the Transformer model.

        :param tokenizer:  A tokenizer with .encode() / .decode() methods
        :param prompt:     The initial string prompt
        :param max_length: Maximum tokens to generate
        :param eos_token_id: Stop if this token is generated
        :param device:     'cpu' or 'cuda'
        :return:           Generated text as a string
        """
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = []

        for _ in range(max_length):
            next_token_id, _ = self.predict_next_token(input_tensor)
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            generated_ids.append(next_token_id)

            # Append the new token to the sequence
            new_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, new_token_tensor], dim=1)

        # Decode to string
        return tokenizer.decode(generated_ids, out_type=str)

class PositionalEncoding(nn.Module):
    """
    Interleaved sinusoidal positional embeddings, applied
    after the token embeddings to inject positional info.
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a (max_len x d_model) matrix of positional embeddings
        pe = torch.zeros(max_len, d_model)

        # position: shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # div_term: shape (d_model/2,) – for sine/cosine frequency scaling
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # Fill even indices with sin, odd indices with cos
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        # Add a batch dimension (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer => not a parameter; stored in state_dict
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model) with positional encodings added
        """
        seq_len = x.size(1)
        # Add positional embeddings up to the seq_len
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)