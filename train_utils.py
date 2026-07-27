import jax
from flax import nnx
from typing import Callable
import grain
import optax
import logging
import dataclasses
import jax.numpy as jnp
import uuid
import os

from tensorboard.summary import Writer as SummaryWriter

TENSORBOARD_DIRECTORY_FMT = 'training_runs/id_{0}'
CHECKPOINTS_DIRECTORY_FMT = 'checkpoints/id_{0}/'


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    model: nnx.Module
    train_data_loader: grain.DataLoader
    eval_data_loader: grain.DataLoader
    optimizer: nnx.Optimizer
    batch_size: int = 100
    training_steps: int = 100
    eval_frequency: int = 10
    run_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4())[:8])
    checkpoint_frequency: int = 10
    checkpoint_path: str = ''
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = optax.softmax_cross_entropy



def train_step(model: nnx.Module,
               optimizer: nnx.Optimizer,
               data: jax.Array,
               labels: jax.Array,
               loss_fn: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
    """Computes a train step on an input batch of data.

    Args:
        model: Jax module to run.
        optimizer: Optimizer to use for executing gradient step.
        data: Input data to feed to the model.
        labels: Labels which to compare model predictions to.
        loss_fn: Loss function on which to evaluate the model. The function should
            calculate loss for each example in the batch.

    Returns:
        Loss function aggregated over the entire batch.
    """
    def compute_loss(model):
        preds = model(data)
        return jnp.mean(loss_fn(preds, labels))

    loss, grads = nnx.value_and_grad(compute_loss)(model)
    optimizer.update(model, grads)
    return loss



def eval_step(model: nnx.Module,
              data: jax.Array,
              labels: jax.Array,
              eval_metric: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
    preds = model(data)
    metrics = eval_metric(preds, labels)
    return jnp.mean(metrics, axis=0)


def basic_train_loop(config: TrainingConfig):
    file_path = TENSORBOARD_DIRECTORY_FMT.format(config.run_id)
    print(
        f"Writing tensorboard output to {file_path}. "
        f"You can launch tensorboard using 'tensorboard --logdir {file_path}'."
    )

    train_writer = SummaryWriter(os.path.join(file_path, 'train'))
    eval_writer = SummaryWriter(os.path.join(file_path, 'test'))

    try:
        for step in range(1, config.training_steps + 1):
            data, labels = next(config.train_data_loader)
            loss = train_step(config.model, config.optimizer, data, labels, config.loss_fn)
            train_writer.add_scalar('Training Loss', loss.item(), step)
            print(f'Step {step}: Training loss - {loss.item():.4f}')

            if step % 10 == 0:
                train_writer.flush()
                eval_writer.flush()

            if step % config.eval_frequency == 0:
                data, labels = next(config.eval_data_loader)
                loss = eval_step(config.model, data, labels, config.loss_fn)
                eval_writer.add_scalar('Eval loss', loss.item(), step)
                print(f'Step {step}: Eval loss - {loss.item():.4f}')
    
    finally:
        train_writer.close()
        eval_writer.close()