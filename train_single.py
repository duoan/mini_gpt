import os
import sys
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import logging

log_filename = f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(log_filename),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Print to the console
    ],
)

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("flax").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import jax
import jax.numpy as jnp
import optax
import jax.random as random
from flax.training import train_state
from model import CausalGPT
from data import CharTokenizer, TokenDataset, jnp_collate_fn
from summary import print_model_summary
from torch.utils.data import DataLoader
import torch
from dataclasses import dataclass

torch.manual_seed(0)

tokenizer = CharTokenizer().load("./tokenizer.json")


# Training config
@dataclass
class Config:
    batch_size = 16
    seq_length = 128
    learning_rate = 5e-4
    num_epochs = 10
    embed_dim = 512
    num_heads = 4
    num_layers = 8
    mlp_dim = 128
    dropout_rate = 0.1
    alibi_bias = True
    dtype = jnp.bfloat16
    param_dtype = jnp.bfloat16
    log_every = 50
    eval_every = 100


config = Config()


def create_train_state(key, model, learning_rate, batch_size, seq_length):
    input_shape = (batch_size, seq_length)
    print_model_summary(model, (seq_length,), batch_size, True)
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
            logger.info(f"Evaluation step {step} / {len(valid_data)}, loss: {loss}")

    return jnp.mean(jnp.array(losses))


def generate(state, input_ids, max_length=10, seq_length=1024):
    for _ in range(max_length):
        logits = state.apply_fn(state.params, input_ids[:, -seq_length:])
        next_token = jnp.argmax(logits, axis=-1)[:, -1]
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=-1)
    context = tokenizer.decode(input_ids[0][:seq_length])
    response = tokenizer.decode(input_ids[0][-max_length:])
    logger.info(f"context:\n{context}")
    logger.info(f"generate:\n{response}")


# Initialize model and state
key = random.PRNGKey(0)
model = CausalGPT(
    vocab_size=tokenizer.vocab_size(),
    embed_dim=config.embed_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    mlp_dim=config.mlp_dim,
    dropout_rate=config.dropout_rate,
    alibi_bias=config.alibi_bias,
    dtype=config.dtype,
    param_dtype=config.param_dtype,
)

state = create_train_state(
    key, model, config.learning_rate, config.batch_size, config.seq_length
)


# Generate dummy data for testing
key, train_key, eval_key = random.split(key, 3)

train_data_loader = DataLoader(
    TokenDataset("tokens_train.npy", config.seq_length),
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=jnp_collate_fn,
)

logger.info(f"train_data_loader size: {len(train_data_loader):,}")

valid_data_loader = DataLoader(
    TokenDataset("tokens_valid.npy", config.seq_length),
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=jnp_collate_fn,
)

logger.info(f"valid_data_loader size: {len(valid_data_loader):,}")

total_steps = config.num_epochs * len(train_data_loader)
# Training loop with evaluation
for epoch in range(config.num_epochs):
    for step, batch in enumerate(train_data_loader):
        state, train_loss = train_step(state, batch)
        global_step = epoch * len(train_data_loader) + step
        if (1 + global_step) % config.log_every == 0:
            logger.info(
                f"Training epoch:{epoch} / {config.num_epochs}, "
                f"Step:{global_step:,}/ {total_steps}, "
                f"Train Loss:{train_loss}"
            )
            generate(state, batch[0], seq_length=config.seq_length)

        if (1 + global_step) % config.eval_every == 0:
            eval_loss = evaluate(state, valid_data_loader)
            logger.info(
                f"Training epoch:{epoch} / {config.num_epochs}, "
                f"Step:{global_step:,}/ {total_steps}, "
                f"Train Loss:{train_loss}"
                f"Eval Loss:{eval_loss}"
            )

logger.info("Training completed!")
