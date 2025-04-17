import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

import sentencepiece as spm
from dataset import TextDataset
from rnn import VanillaRNNLanguageModel
from lstm import LSTMLanguageModel
from transformer import TransformerLanguageModel


# ----------  CONSTANTS / PATHS  ------------------
ORIGINAL_TRAIN_FILE = "data/train.jsonl"
TRAIN_SPLIT_FILE    = "data/train_split.jsonl"
VAL_SPLIT_FILE      = "data/val_split.jsonl"
TEST_FILE           = "data/test.jsonl"
TOKENIZER_PATH      = "bpe_tokenizer.model"

MAX_SEQ_LEN  = 256
BATCH_SIZE   = 128
EMBED_DIM    = 256
HIDDEN_DIM   = 512
NUM_LAYERS   = 3
EPOCHS       = 30
LEARNING_RATE= 1e-3

# ----------  SPLIT / TOKENIZER UTILS  -----------
def split_train_file(original_file=ORIGINAL_TRAIN_FILE, train_file=TRAIN_SPLIT_FILE, val_file=VAL_SPLIT_FILE, train_ratio=0.8):
    """
    Splits the original train.jsonl into two files: train_split.jsonl and val_split.jsonl
    according to a given train_ratio (e.g., 0.8 for 80%).
    """
    with open(original_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)  # Randomize order before splitting

    split_idx = int(train_ratio * len(lines))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Write train portion
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)

    # Write validation portion
    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)

def load_tokenizer(model_file):
    """
    Load a trained SentencePiece tokenizer from a .model file.

    :param model_file: The path to the SentencePiece model file (e.g. bpe_tokenizer.model).
    :return: A SentencePieceProcessor instance for encoding/decoding text.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp

# ----------  COLLATE / DATA UTILS  -------------
def collate_fn(batch):
    """
    Ensure batch is appropriately sized and padded for efficient training

    :param batch: batch from DataLoader, which will be a list of Tuples of token ID tensors (which
        could be different sizes)
    :return: collated input and target batch
    """
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(
        input_batch, batch_first=True, padding_value=3
    )
    target_batch = nn.utils.rnn.pad_sequence(
        target_batch, batch_first=True, padding_value=3
    )
    return input_batch, target_batch

def compute_token_bleu(model, jsonl_file, tokenizer, device):
    references = []
    candidates = []
    smoothing_fn = SmoothingFunction().method1

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(tqdm(lines)):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]

        # Generate only 1 token
        input_ids = tokenizer.encode(prompt_text, out_type=int)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        token_id, _ = model.predict_next_token(input_tensor)
        generated_token = tokenizer.decode([token_id]).strip()
        reference_token = reference_text.strip()

        candidates.append(nltk.word_tokenize(generated_token))
        references.append([nltk.word_tokenize(reference_token)])

    bleu = corpus_bleu(references, candidates, smoothing_function=smoothing_fn)
    print(f"\n[Token-Level] BLEU Score: {bleu:.4f}")
    return bleu

# ----------  METRIC: BLEU  ----------------------
def compute_bleu_from_jsonl(model, jsonl_file, tokenizer, device, max_gen_len=50):
    """
    Computes BLEU by reading raw prompt/completion pairs from test.jsonl,
    generating text from the model for each prompt, and comparing to reference completions.

    :param model: Your trained language model
    :param jsonl_file: Path to e.g. "data/test.jsonl"
    :param tokenizer: A SentencePiece tokenizer instance
    :param device: torch.device("cpu") or torch.device("cuda")
    :param max_gen_len: Max tokens to generate for each prompt
    :return: BLEU score (float)
    """
    references = []   # Will store list of list(s) of reference tokens
    candidates = []   # Will store list of candidate tokens

    model.eval()
    smoothing_fn = SmoothingFunction().method1  # use a simple smoothing method

    # Read each example from the JSONL
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]

        # Generate text from the model
        generated_text = model.generate(
            prompt=prompt_text,
            tokenizer=tokenizer,
            max_length=max_gen_len,
            device=device
        )

        # Tokenize both reference and candidate in word space
        reference_tokens = nltk.word_tokenize(reference_text)
        candidate_tokens = nltk.word_tokenize(generated_text)

        # Add them to the big lists
        references.append([reference_tokens])
        candidates.append(candidate_tokens)

    # Compute corpus-level BLEU with smoothing
    bleu_score = corpus_bleu(
        references,
        candidates,
        smoothing_function=smoothing_fn
    )

    return bleu_score

# ----------  METRIC: PERPLEXITY  ----------------
def evaluate_perplexity(model, test_loader, vocab_size, device, pad_token_id=3):
    """
    Compute the perplexity of the model on the test set.

    :param model: Your trained language model (vanilla RNN, LSTM, or Transformer)
    :param test_loader: DataLoader for the test dataset
    :param vocab_size: The size of the tokenizer vocabulary
    :param device: 'cuda' or 'cpu'
    :param pad_token_id: ID used for <pad>, so we can ignore it in the loss
    :return: perplexity (float)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_token_id)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in test_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            # Count non-padding tokens
            num_tokens = (target_ids != pad_token_id).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    return perplexity

# ----------  TRAINING  --------------------------
def train_model(model, train_loader, val_loader, vocab_size, device,
                epochs=EPOCHS, lr=LEARNING_RATE, early_stopping_patience=3):
    """
    A generic train function that can handle either RNN, LSTM, or Transformer.
    """
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=1, factor=0.5, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    best_val_loss = float("inf")
    no_improve_epochs = 0

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # ----- Training -----
        model.train()
        total_train_loss = 0.0

        for input_ids, target_ids in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"
        ):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for input_ids, target_ids in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"
            ):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(
                    f"No improvement for {early_stopping_patience} epochs. Stopping early."
                )
                break

    print("Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    return model, train_losses, val_losses

# ----------  MAIN SCRIPT  -----------------------
def main(model_type="rnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split data
    split_train_file()

    # Load tokenizer, create dataset
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()

    train_dataset = TextDataset(TRAIN_SPLIT_FILE, tokenizer, MAX_SEQ_LEN)
    val_dataset = TextDataset(VAL_SPLIT_FILE, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Instantiate the chosen model
    if model_type.lower() == "rnn":
        print(">>> Using VanillaRNNLanguageModel...")
        model = VanillaRNNLanguageModel(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        ).to(device)
    elif model_type.lower() == "lstm":
        print(">>> Using LSTMLanguageModel...")
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
        ).to(device)
    elif model_type.lower() == "transformer":
        print(">>> Using TransformerLanguageModel...")
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,    
            nhead=4,               
            num_layers=NUM_LAYERS, 
            feedforward_dim=512,   
            dropout=0.2,           
            pad_token_id=3,        
            max_seq_len=512        
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'rnn' or 'lstm' or 'transformer.")

    # Train the model
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        vocab_size,
        device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        early_stopping_patience=3
    )

    # Save loss curves
    loss_data = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }

    with open(f"loss_data/{model_type}_loss.json", "w") as f:
        json.dump(loss_data, f)

    # # Plot the train/val loss curves
    # epochs_range = range(1, len(train_losses) + 1)
    # plt.figure()
    # plt.plot(epochs_range, train_losses, label="Train Loss")
    # plt.plot(epochs_range, val_losses, label="Val Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training and Validation Loss")
    # plt.legend()
    # plt.show()

    # Save the trained model
    filename = f"models/{model_type}_e{EMBED_DIM}_h{HIDDEN_DIM}_l{NUM_LAYERS}.pt"
    torch.save(model.state_dict(), filename)
    print(f"Saved model to: {filename}")

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate on test set
    test_dataset = TextDataset(TEST_FILE, tokenizer, MAX_SEQ_LEN)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)
    
    # --- Perplexity ---
    ppl = evaluate_perplexity(model, test_loader, vocab_size, device)
    print(f"Test Perplexity: {ppl:.3f}")

    # --- BLEU ---
    bleu = compute_bleu_from_jsonl(model, TEST_FILE, tokenizer, device, max_gen_len=1)
    print(f"Test BLEU: {bleu:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="rnn",
                        choices=["rnn","lstm", "transformer"],
                        help="Which model to train: 'rnn' or 'lstm' or 'transformer")
    args = parser.parse_args()

    main(model_type=args.model_type)