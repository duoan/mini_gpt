import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.random as random
from flax.training import train_state
from model import CausalGPT


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
def train_step_pmap(state, batch):
    key, dropout_key = random.split(state.key)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch,
            deterministic=False,
            rngs={"dropout": dropout_key},
        )
        loss = jnp.mean(jnp.square(logits))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(key=key)
    return state, loss


@jax.jit
def eval_step_pmap(state, batch):
    logits = state.apply_fn(
        {"params": state.params},
        batch,
        deterministic=True,
    )
    loss = jnp.mean(jnp.square(logits))
    return loss


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
        eval_every: int = 100,
    ):
        self.model = model
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_epochs = num_epochs
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

    def _prepare_batch(self, batch):
        return batch.reshape(self.num_devices, self.device_batch_size, self.seq_length)

    def train_epoch(self, train_data, eval_data=None):
        for step, batch in enumerate(train_data):
            p_batch = self._prepare_batch(batch)
            self.state, train_loss = self.p_train_step(self.state, p_batch)
            train_loss = jnp.mean(train_loss)

            if eval_data is not None and step % self.eval_every == 0:
                eval_losses = []
                for eval_batch in eval_data:
                    p_eval_batch = self._prepare_batch(eval_batch)
                    eval_loss = self.p_eval_step(self.state, p_eval_batch)
                    eval_losses.append(jnp.mean(eval_loss))
                eval_loss = jnp.mean(jnp.array(eval_losses))
                print(
                    f"Step {step}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}"
                )


if __name__ == "__main__":
    model = CausalGPT(
        vocab_size=50257,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        dropout=0.1,
    )

    batch_size = 32
    seq_length = 128
    trainer = DataParallelTrainer(
        model=model,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    key = random.PRNGKey(0)
    train_data = [
        random.randint(key, (batch_size, seq_length), 0, 50257) for _ in range(10)
    ]
    eval_data = [
        random.randint(key, (batch_size, seq_length), 0, 50257) for _ in range(2)
    ]

    for epoch in range(trainer.num_epochs):
        print(f"Epoch {epoch}")
        trainer.train_epoch(train_data, eval_data)
