import os

from sympy import im

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.random as random
from flax.training import train_state
from model import CausalGPT
from data import TokenDataset, jnp_collate_fn
from generate import generate_text, generate_from_batch
from torch.utils.data import DataLoader
import torch
import tiktoken

torch.manual_seed(0)

encoding = tiktoken.get_encoding("o200k_base")


class TrainState(train_state.TrainState):
    key: jnp.ndarray


def create_train_state(key, model, learning_rate, batch_size, seq_length, num_devices):
    input_shape = (num_devices, batch_size // num_devices, seq_length)
    variables = model.init(key, jnp.ones(input_shape[1:], dtype=jnp.int32))

    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        key=key,
    )


@jax.jit
def train_step_pmap(state, ids, labels):
    key, dropout_key = random.split(state.key)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            ids,
            deterministic=False,
            rngs={"dropout": dropout_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(key=key)
    return state, loss


@jax.jit
def eval_step_pmap(state, ids, labels):
    logits = state.apply_fn(
        {"params": state.params},
        ids,
        deterministic=True,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)


def generate(state, input_ids, max_length=10, seq_length=1024):
    for _ in range(max_length):
        logits = state.apply_fn(state.params, input_ids[:, -seq_length:])
        next_token = jnp.argmax(logits, axis=-1)[:, -1]
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=-1)
    print("context:\n", encoding.decode(input_ids[0][:seq_length]))
    print("generate:\n", encoding.decode(input_ids[0][-max_length:]))


class DataParallelTrainer:
    """
    Data parallelism, as the name suggests, focuses on parallelizing the data processing of the model.
    If we are given a very large batch, we divide the batch into smaller batches and distribute them across multiple devices.
    Each device will process a different batch of data in parallel.
    Afterwards, we will aggregate the results from each device to update the model.
    Data parallelism is the most common parallelism strategy used in deep learning and well supported in most frameworks
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        seq_length: int,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        log_every: int = 10,
        eval_every: int = 100,
    ):
        self.model = model
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_epochs = num_epochs
        self.log_every = log_every
        self.eval_every = eval_every

        self.num_devices = jax.device_count()
        assert batch_size % self.num_devices == 0
        self.device_batch_size = batch_size // self.num_devices

        self.key = random.PRNGKey(0)
        self.key = random.split(self.key, num=self.num_devices)

        self.state = create_train_state(
            self.key[0], model, learning_rate, batch_size, seq_length, self.num_devices
        )

        self.p_train_step = jax.pmap(train_step_pmap, axis_name="batch")
        self.p_eval_step = jax.pmap(eval_step_pmap, axis_name="batch")
        self.state = jax.device_put_replicated(self.state, jax.devices())

    def _prepare_batch(self, ids, labels):
        return (
            ids.reshape(self.num_devices, self.device_batch_size, self.seq_length),
            labels.reshape(self.num_devices, self.device_batch_size, self.seq_length),
        )

    def train_epoch(self, train_data, eval_data=None):
        for step, (ids, labels) in enumerate(train_data):
            p_ids, p_labels = self._prepare_batch(ids, labels)
            self.state, train_loss = self.p_train_step(self.state, p_ids, p_labels)
            train_loss = jnp.mean(train_loss)
            if step % self.log_every == 0:
                print(f"Step {step}, Train Loss: {train_loss:.4f}")

            if eval_data is not None and step % self.eval_every == 0:
                eval_losses = []
                for ids, labels in eval_data:
                    p_ids, p_labels = self._prepare_batch(ids, labels)
                    eval_loss = self.p_eval_step(self.state, p_ids, p_labels)
                    eval_losses.append(jnp.mean(eval_loss))
                eval_loss = jnp.mean(jnp.array(eval_losses))
                print(
                    f"Step {step}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}"
                )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """Generate text using the trained model"""
        return generate_text(
            self.state,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seq_length=self.seq_length,
            encoding=encoding,  # 使用全局的encoding
        )


if __name__ == "__main__":
    model = CausalGPT(
        vocab_size=encoding.n_vocab,
        embed_dim=768,
        num_heads=16,
        num_layers=4,
        mlp_dim=512,
        dropout=0.1,
        dtype=jnp.bfloat16,
    )

    batch_size = jax.device_count() * 4
    seq_length = 512
    trainer = DataParallelTrainer(
        model=model,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    train_data_loader = DataLoader(
        TokenDataset("tokens_train.npy", seq_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=jnp_collate_fn,
    )

    print(f"train_data_loader size: {len(train_data_loader):,}")

    valid_data_loader = DataLoader(
        TokenDataset("tokens_valid.npy", seq_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=jnp_collate_fn,
    )

    print(f"valid_data_loader size: {len(valid_data_loader):,}")

    for epoch in range(trainer.num_epochs):
        print(f"Epoch {epoch}")
        trainer.train_epoch(train_data_loader, valid_data_loader)
