from typing import Tuple
import os
import sentencepiece as spm

def add_special_tokens(pairs: Tuple[list]):
    """
    Insert <bos> and <eos> special tokens into a dataset
    :param: pairs: original prompts and completions
    :return: prompts/completion pairs with special tokens inserted
    """
    new_prompts = []
    new_completions = []

    for prompt, completion in zip(pairs):
        # If the beginning of the prompt is upper case, then we assume it is the start of a sequence
        if prompt[0].isupper():
            prompt = '<bos>' + prompt

        # If the end of the completion is a terminating punctuation, then we assume it is the end of a sequence
        if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
            completion += '<eos>'
        new_prompts.append(prompt)
        new_completions.append(completion)

    return new_prompts, new_completions

# Merge all text files into a single corpus 
def merge_text_files(data_dir, output_file):
    """
    This will merge all textual data in a directory into a single corpus
    :param data_dir: path to the directory containing the raw text files
    :param output_file: path to file where corpus will be saved
    """
    # open new file
    with open(output_file, "w", encoding="utf-8") as outfile:
        # WRITE CODE FOR THIS
        print("opening file")

    
if __name__ == "__main__":
    DATA_DIR = "./data/raw" # path to raw data directory
    TOKENIZER_PREFIX = "bpe_tokenizer" # this will be used for naming the tokenizer
    VOCAB_SIZE = 10000 # stopping condition for tokenizing
    CORPUS_FILE = "corpus.txt" # path to new combined corpus file
    merge_text_files(DATA_DIR, CORPUS_FILE)

    # Train the tokenizer with special tokens
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        bos_id=1, # this is set to 1 because 0 is <unk>
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"])
    )

    print("Tokenizer training complete! Files generated:")
    print(f"- {TOKENIZER_PREFIX}.model")
    print(f"- {TOKENIZER_PREFIX}.vocab")