import torch
from rnn import VanillaRNNLanguageModel
from dataset import TextDataset
from torch.utils.data import DataLoader
from main import load_tokenizer, evaluate_perplexity, compute_bleu_from_jsonl, collate_fn, TOKENIZER_PATH, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, TEST_FILE

if __name__ == "__main__":
    #############################
    # 1) Load the Trained Model #
    #############################
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize a model of the same class/architecture as what was trained
    model = VanillaRNNLanguageModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    # Load the saved weights
    model.load_state_dict(torch.load("rnn_final_model.pt", map_location=device))
    model.eval()

    ##############################
    # 2) Create the Test Dataset #
    ##############################
    test_dataset = TextDataset("data/test.jsonl", tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    #############################
    # 3) Evaluate Perplexity    #
    #############################
    ppl = evaluate_perplexity(model, test_loader, vocab_size, device)
    print(f"Test Perplexity: {ppl:.3f}")

    #############################
    # 4) Evaluate BLEU          #
    #############################
    bleu = compute_bleu_from_jsonl(model, TEST_FILE, tokenizer, device)
    print(f"Test BLEU: {bleu:.4f}")