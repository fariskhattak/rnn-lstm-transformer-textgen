import argparse
import torch
import sentencepiece as spm

from rnn import VanillaRNNLanguageModel
from lstm import LSTMLanguageModel
from transformer import TransformerLanguageModel

# ---------- SETTINGS ----------
TOKENIZER_PATH = "bpe_tokenizer.model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN_ID = 3  # consistent with training

# ---------- LOAD TOKENIZER ----------
def load_tokenizer(path):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

# ---------- LOAD MODEL ----------
def load_model(model_type, model_path, vocab_size, embed_dim, hidden_dim, num_layers):
    if model_type == "rnn":
        model = VanillaRNNLanguageModel(
            vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, pad_token_id=PAD_TOKEN_ID)
    elif model_type == "lstm":
        model = LSTMLanguageModel(
            vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, pad_token_id=PAD_TOKEN_ID)
    elif model_type == "transformer":
                model = TransformerLanguageModel(vocab_size=vocab_size, embed_dim=embed_dim, nhead=4, num_layers=num_layers, feedforward_dim=512, dropout=0.2, pad_token_id=PAD_TOKEN_ID)

    else:
        raise ValueError("Invalid model type. Choose from 'rnn', 'lstm', 'transformer'")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ---------- CHAT LOOP ----------
def chat(model, tokenizer):
    print("\n>>> Chat session started. Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() in {"exit", "quit"}:
            print("Session ended.")
            break
        response = model.generate(tokenizer=tokenizer, prompt=prompt, max_length=50, device=DEVICE)
        print("Bot:", response)

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="rnn", choices=["rnn", "lstm", "transformer"],
                        help="Choose model type to chat with")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model (.pt) file")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding dimension size")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer size (ignored for Transformer FFN)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of RNN/LSTM/Transformer layers")
    args = parser.parse_args()

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()
    model = load_model(
        args.model_type, args.model_path, vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    chat(model, tokenizer)

if __name__ == "__main__":
    main()
