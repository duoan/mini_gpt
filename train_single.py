import jax
import jax.numpy as jnp
import optax
import jax.random as random
from flax.training import train_state
from model import CausalGPT


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
    def loss_fn(params):
        logits = state.apply_fn(params, batch)
        loss = jnp.mean(jnp.square(logits))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch)
    loss = jnp.mean(jnp.square(logits))
    return loss


def evaluate(state, eval_ds):
    losses = []
    for batch in eval_ds:
        loss = eval_step(state, batch)
        losses.append(loss)
    return jnp.mean(jnp.array(losses))


# Training setup
batch_size = 8
seq_length = 128
learning_rate = 1e-3
num_epochs = 10
eval_every = 10

# Initialize model and state
key = random.PRNGKey(0)
model = CausalGPT(
    vocab_size=50257,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    mlp_dim=3072,
    dropout=0.1,
)

state = create_train_state(key, model, learning_rate, batch_size, seq_length)

# Generate dummy data for testing
key, train_key, eval_key = random.split(key, 3)
train_data = [
    random.randint(train_key, (batch_size, seq_length), 0, 50257) for _ in range(10)
]
eval_data = [
    random.randint(eval_key, (batch_size, seq_length), 0, 50257) for _ in range(2)
]

# Training loop with evaluation
for epoch in range(num_epochs):
    for step, batch in enumerate(train_data):
        state, train_loss = train_step(state, batch)

        if (1 + step) % eval_every == 0:
            eval_loss = evaluate(state, eval_data)
            print(
                f"Epoch {epoch}, Step {step}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}"
            )

print("Training completed!")
