# Language Modeling with RNN, LSTM, and Transformer

This project implements and compares three types of neural language models — Vanilla RNN, LSTM, and Transformer — for next-token prediction using a corpus of literary text. The models are trained and evaluated on tokenized inputs with a SentencePiece BPE tokenizer.

---

## Features
- Token-level language modeling with RNN, LSTM, and Transformer
- SentencePiece tokenizer with BPE encoding
- Evaluation using **Perplexity** and **BLEU score**
- Nucleus sampling (`top-p`) for text generation
- CLI for training, testing, and chatting with models

---

## Project Structure

```
.
├── data/                   # JSONL datasets (train.jsonl, val.jsonl, test.jsonl)
├── models/                 # Trained model weights
├── loss_data/              # Trained model loss logs
├── rnn.py                  # RNN model implementation
├── lstm.py                 # LSTM model implementation
├── transformer.py          # Transformer model implementation
├── dataset.py              # TextDataset class for loading and batching
├── main.py                 # Training script
├── test.py                 # Evaluation script
├── chat.py                 # Chat CLI to interact with a trained model
├── README.md               # This file
└── bpe_tokenizer.model     # Trained SentencePiece tokenizer model
```

---

## Requirements

Install the necessary libraries:
> You’ll need PyTorch, tqdm, sentencepiece, matplotlib, and nltk.

Also run:
```python
import nltk
nltk.download('punkt')
```

---

## Training a Model

```bash
python main.py --model_type rnn         # or lstm or transformer
```
Specifications can be edited within the code of `main.py`

Trained model checkpoints and loss logs will be saved to `models/`.

---

## Evaluation

```bash
python test.py
```

This will:
- Load the saved model from `models/`
- Compute **Perplexity** and **BLEU score**

---

## Chat with a Model

You can interact with any **trained model** via CLI by specifying the model type and path to the `.pt` file. You can also optionally customize the architecture using `--embed_dim`, `--hidden_dim`, and `--num_layers`.

### RNN Example
```bash
python chat.py \
  --model_type rnn \
  --model_path models/rnn_e256_h512_l2.pt
  --embed_dim 256 \
  --hidden_dim 512 \
  --num_layers 2
```

### LSTM Example
```bash
python chat.py \
  --model_type lstm \
  --model_path models/lstm_e256_h512_l3.pt \
  --embed_dim 256 \
  --hidden_dim 512 \
  --num_layers 3
```

### Transformer Example
```bash
python chat.py \
  --model_type transformer \
  --model_path models/transformer_e512_hNone_l6.pt \
  --embed_dim 512 \
  --num_layers 6
```

Example:
```
You: Tell me a story about the moon
Bot: The moon rose slowly over the silent sea, casting silver light on the waves.
```

---

## Visualizations

To plot training and validation loss curves:
```bash
python plot_losses.py   # Requires that loss history was saved during training
```
Edit which loss plot to view in the code of `plot_losses.py`

Note: Loss data is based on these models:

1. `transformer_e512_hNone_l6.pt`
2. `lstm_e256_h512_l3.pt`
3. `rnn_e256_h512_l2.pt`

---

## Results Summary

| Model       | Perplexity | BLEU Score | Best Validation Loss |
|-------------|------------|------------|----------------------|
| RNN         | 102.011    | 0.0003     | 4.6333
| LSTM        | 101.874    | 0.0001     | 4.6280
| Transformer | 82.228     | 0.0003     | 4.4270

---

---

## Author
Created by Faris Khattak for CSC 4700 Foundational AI Project 2.
