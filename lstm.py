import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.2, pad_token_id=0):
        """
        Create a Vanilla Recurrent Neural Network Language Model

        :param vocab_size: size of the vocabulary
        :param embed_dim: size of each token's embedding vector
        :param hidden_dim: size of the RNN hidden states 
        :param num_layers: number of RNNs to stack
        :param dropout: training dropout rates
        :param pad_token_id: token ID of <pad> token
        """
        super(LSTMLanguageModel, self).__init__()
        # Define embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        # Define LSTM layer
        self.lstm = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # Output layer that maps hidden state to output
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        Compute model output logits given a sequence

        :param input_ids: Sequence of input token IDs, Tensor of shape (batch_size, seq_len)
        :param hidden: previous hidden state
        :return: output logits, Tensor of shape (batch_size, seq_len, vocab_size)
        """
        embeds = self.embedding(input_ids) # compute embeddings for all input tokens in parallel
        output, hidden = self.lstm(embeds, hidden) # pass embeddings through the RNN layers
        logits = self.fc(output) # compute output logits
        return logits, hidden
    
    def predict_next_token(self, input_ids):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids

        :param input_ids: Input sequence token IDs
        :return: next token ID, hidden state
        """
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids)
            logits = logits[:, -1:]
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)
            return next_token_id.item(), hidden

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, device='cuda'):
        """
        Generate a full output sequence given a prompt

        :param tokenizer: The trained SentencePiece tokenizer
        :param prompt: The input prompt (plain text string)
        :param max_length: Maximum number of tokens to generate autoregressively before stopping
        :param eos_token_id: The token ID of the EOS token
        :param device: Device we are using to run the model
        """
        self.eval() # set the model to evaluation mode
        input_ids = tokenizer.encode(prompt, out_type=int) # Encode the input string into token IDs
        # convert token ID list to tensor, move to device memory, and adding a batch dimension (1D to 2D)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        generate_ids = [] # this will store the generated token IDs
        hidden = None # initial hidden state is None

        # loop over max output tokens
        for _ in range(max_length):
            next_token_id, hidden = self.predict_next_token(input_tensor)
            # exit early if the model generated <eos> token ID
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            # keep track of generated tokens
            generate_ids.append(next_token_id)
            # the input to the next step is just the new token and the hidden state
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        #decode generated token IDs into tokens
        return tokenizer.decode(generate_ids, out_type=str)