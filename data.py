import os
import tiktoken
import jax.numpy as jnp
from torch.utils.data import Dataset


def preprocess():
    text = ""
    for txt_file in os.listdir("./input"):
        with open("./input/" + txt_file, "r", encoding="utf-8") as f:
            text += f.read()

    print(f"Length of dataset in characters: {len(text):,}")

    encoding = tiktoken.get_encoding("o200k_base")
    print(f"Length of vocab: {encoding.n_vocab:,}")

    tokens = encoding.encode(text)

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
