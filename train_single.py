import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import optax
import jax.random as random
from flax.training import train_state
from model import CausalGPT
from data import CharTokenizer, TokenDataset, jnp_collate_fn
from torch.utils.data import DataLoader
import torch

torch.manual_seed(0)

tokenizer = CharTokenizer().load("./tokenizer.json")

# Training setup
batch_size = 16
seq_length = 128
learning_rate = 5e-4
num_epochs = 10
log_every = 50
eval_every = 100


def create_train_state(key, model, learning_rate, batch_size, seq_length):
    input_shape = (batch_size, seq_length)
    params = model.init(key, jnp.ones(input_shape, dtype=jnp.int32))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@jax.jit
def train_step(state, batch):
    ids, labels = batch

    def loss_fn(params):
        logits = state.apply_fn(params, ids)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    ids, labels = batch
    logits = state.apply_fn(state.params, ids)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)


def evaluate(state, valid_data):
    losses = []
    for step, batch in enumerate(valid_data):
        loss = eval_step(state, batch)
        losses.append(loss)
        if step % 100 == 0:
            print(f"Evaluation step {step} / {len(valid_data)}, loss: {loss:6f}")

    return jnp.mean(jnp.array(losses))


def generate(state, input_ids, max_length=10, seq_length=1024):
    for _ in range(max_length):
        logits = state.apply_fn(state.params, input_ids[:, -seq_length:])
        next_token = jnp.argmax(logits, axis=-1)[:, -1]
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=-1)
    print("context:\n", tokenizer.decode(input_ids[0][:seq_length]))
    print("generate:\n", tokenizer.decode(input_ids[0][-max_length:]))


# Initialize model and state
key = random.PRNGKey(0)
model = CausalGPT(
    vocab_size=tokenizer.vocab_size(),
    embed_dim=512,
    num_heads=4,
    num_layers=4,
    mlp_dim=128,
    dropout=0.1,
    dtype=jnp.float32,
)

state = create_train_state(key, model, learning_rate, batch_size, seq_length)

# Generate dummy data for testing
key, train_key, eval_key = random.split(key, 3)

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

total_steps = num_epochs * len(train_data_loader)
# Training loop with evaluation
for epoch in range(num_epochs):
    for step, batch in enumerate(train_data_loader):
        state, train_loss = train_step(state, batch)
        global_step = epoch * len(train_data_loader) + step
        if (1 + global_step) % log_every == 0:
            print(
                f"Training epoch:{epoch} / {num_epochs}, "
                f"Step:{global_step:,}/ {total_steps}, "
                f"Train Loss:{train_loss:6f}"
            )
            generate(state, batch[0], seq_length=seq_length)

        if (1 + global_step) % eval_every == 0:
            eval_loss = evaluate(state, valid_data_loader)
            print(
                f"Training epoch:{epoch} / {num_epochs}, "
                f"Step:{global_step:,}/ {total_steps}, "
                f"Train Loss:{train_loss:6f}"
                f"Eval Loss:{eval_loss:6f}"
            )

print("Training completed!")
