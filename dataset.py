import json
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=128):
        """
        Create a text dataset for PyTorch that handles JSONL data
        for causal language modeling.

        :param filepath: path to the JSONL file
        :param tokenizer: instance of a trained SentencePiece tokenizer
        :param max_seq_len: maximum sequence length to allow
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Read each line from the JSONL file and tokenize
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Combine prompt + completion for causal LM
                text = item["prompt"] + " " + item["completion"]

                # Tokenize, then truncate to max_seq_len
                token_ids = tokenizer.encode(text, out_type=int)[:max_seq_len]

                # We skip very short samples (need at least 2 tokens for input vs. target)
                if len(token_ids) < 2:
                    continue

                self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get and format a sample at the given index.

        For causal language modeling, we train the model to predict every
        next token in the sequence given the prior ones.

        :param idx: index of the sample
        :return: (input_ids, target_ids) pair
        """
        tokens = self.samples[idx]
        # Input is everything except the last token
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        # Target is everything except the first token
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids
    
    def get_raw_data(self, idx):
        """
        Return the raw prompt and completion for the sample at 'idx'.
        This is used by compute_bleu for reference text.
        """
        return self.raw_data[idx]