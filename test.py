import torch
from torch.utils.data import DataLoader

# Import the model classes
from rnn import VanillaRNNLanguageModel
from lstm import LSTMLanguageModel
from transformer import TransformerLanguageModel

# Import utilities
from dataset import TextDataset
from main import (
    load_tokenizer, 
    evaluate_perplexity, 
    compute_bleu_from_jsonl, 
    collate_fn, 
    TOKENIZER_PATH, 
    EMBED_DIM, 
    HIDDEN_DIM, 
    NUM_LAYERS, 
    BATCH_SIZE, 
    MAX_SEQ_LEN, 
    TEST_FILE
)

def test_model(model_class, model_path, model_name, tokenizer, test_loader, vocab_size, device):
    """
    Helper function to:
      1) Instantiate a model of 'model_class'
      2) Load trained weights from 'model_path'
      3) Evaluate perplexity and BLEU
      4) Print results
    """
    print(f"\nEvaluating {model_name}...")
    
    # 1) Instantiate and load
    if model_name.lower() == "transformer":
        model = model_class(
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
        # For both RNN and LSTM
        model = model_class(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2) Evaluate Perplexity
    ppl = evaluate_perplexity(model, test_loader, vocab_size, device)
    print(f"[{model_name}] Test Perplexity: {ppl:.3f}")

    # 3) Evaluate BLEU
    bleu = compute_bleu_from_jsonl(model, TEST_FILE, tokenizer, device)
    print(f"[{model_name}] Test BLEU: {bleu:.4f}")
    

if __name__ == "__main__":
    # --- Common Setup ---
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the test dataset/loader
    test_dataset = TextDataset(TEST_FILE, tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Paths to each model’s saved weights
    model_paths = {
        "rnn": "models/rnn_final_model.pt",
        "lstm": "models/lstm_final_model.pt",
        "transformer": "models/transformer_final_model.pt"
    }

    # Evaluate all three:
    #  1) Vanilla RNN
    test_model(
        model_class=VanillaRNNLanguageModel,
        model_path=model_paths["rnn"],
        model_name="RNN",
        tokenizer=tokenizer,
        test_loader=test_loader,
        vocab_size=vocab_size,
        device=device
    )

    #  2) LSTM
    test_model(
        model_class=LSTMLanguageModel,
        model_path=model_paths["lstm"],
        model_name="LSTM",
        tokenizer=tokenizer,
        test_loader=test_loader,
        vocab_size=vocab_size,
        device=device
    )

    #  3) Transformer
    test_model(
        model_class=TransformerLanguageModel,
        model_path=model_paths["transformer"],
        model_name="Transformer",
        tokenizer=tokenizer,
        test_loader=test_loader,
        vocab_size=vocab_size,
        device=device
    )
