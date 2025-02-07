import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax.numpy as jnp
from torch.utils.data import Dataset


import json


class CharTokenizer:
    def __init__(self, texts=None, special_tokens=None):
        """Initialize the tokenizer with optional special tokens."""
        self.special_tokens = special_tokens or ["<pad>", "<unk>"]
        self.vocab = {}  # Char -> Index mapping
        self.inv_vocab = {}  # Index -> Char mapping
        if texts:
            self.fit(texts)

    def fit(self, texts):
        """Builds vocabulary from a list of texts."""
        unique_chars = sorted(set("".join(texts)))  # Get all unique characters
        all_tokens = self.special_tokens + unique_chars  # Add special tokens

        self.vocab = {char: idx for idx, char in enumerate(all_tokens)}
        self.inv_vocab = {idx: char for char, idx in self.vocab.items()}

    def encode(self, text):
        """Encodes a string into a list of indices."""
        return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]

    def decode(self, indices):
        """Decodes a list of indices back into a string."""
        return "".join(self.inv_vocab.get(idx, "<unk>") for idx in indices.tolist())

    def vocab_size(self):
        """Returns the vocabulary size."""
        return len(self.vocab)

    def save(self, file_path):
        """Saves the tokenizer vocabulary to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

    def load(self, file_path):
        """Loads the tokenizer vocabulary from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        # Rebuild inverse vocabulary
        self.inv_vocab = {idx: char for char, idx in self.vocab.items()}
        return self


def preprocess():
    text = ""
    for txt_file in os.listdir("./input/harry_potter"):
        with open("./input/harry_potter/" + txt_file, "r", encoding="utf-8") as f:
            text += f.read()

    print(f"Length of dataset in characters: {len(text):,}")

    tokenizer = CharTokenizer([text])
    tokenizer.save("./tokenizer.json")
    print(f"Vocabulary size: {tokenizer.vocab_size():,}")

    tokens = tokenizer.encode(text)

    print(f"Length of dataset in tokens: {len(tokens):,}")

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        f.write(str(tokens))
    n = int(0.9 * len(tokens))
    token_array = jnp.array(tokens, dtype=jnp.int32)
    jnp.save("tokens_train.npy", token_array[:n])
    jnp.save("tokens_valid.npy", token_array[n:])


def jnp_collate_fn(batch):
    transposed_data = list(zip(*batch))
    ids = jnp.array(transposed_data[0])
    labels = jnp.array(transposed_data[1])
    return ids, labels


class TokenDataset(Dataset):
    def __init__(self, path, seq_length):
        with open(path, "rb") as f:
            self.tokens = jnp.load(f)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.tokens[idx : idx + self.seq_length],
            self.tokens[idx + 1 : idx + self.seq_length + 1],
        )


def main():
    preprocess()


if __name__ == "__main__":
    main()
