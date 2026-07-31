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
import data
import jax.profiler
from orbax.checkpoint import v1 as ocp

from tensorboard.summary import Writer as SummaryWriter

TENSORBOARD_DIRECTORY_FMT = 'training_runs/id_{0}'
CHECKPOINTS_DIRECTORY_FMT = 'checkpoints/id_{0}/'


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    model: nnx.Module
    data_config: data.DataConfig
    optimizer: nnx.Optimizer
    batch_size: int = 100
    training_steps: int = 100
    eval_frequency: int = 10
    run_id: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())[:8])
    checkpoint_frequency: int = 5
    loss_fn: Callable[[jax.Array, jax.Array],
                      jax.Array] = optax.softmax_cross_entropy


def make_train_step(
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array]
) -> Callable[[nnx.Module, nnx.Optimizer, jax.Array, jax.Array], jax.Array]:

    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, data: jax.Array,
                   labels: jax.Array) -> jax.Array:
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

    return train_step


def make_eval_step(
    eval_metric: Callable[[jax.Array, jax.Array], jax.Array]
) -> Callable[[nnx.Module, nnx.Optimizer, jax.Array, jax.Array], jax.Array]:

    @nnx.jit
    def eval_step(model: nnx.Module, data: jax.Array,
                  labels: jax.Array) -> jax.Array:
        preds = model(data)
        metrics = eval_metric(preds, labels)
        return jnp.mean(metrics)

    return eval_step


def basic_train_loop(config: TrainingConfig):
    file_path = TENSORBOARD_DIRECTORY_FMT.format(config.run_id)
    checkpoint_path = os.path.abspath(CHECKPOINTS_DIRECTORY_FMT.format(config.run_id))
    print(
        f"Writing tensorboard output to {file_path}. "
        f"You can launch tensorboard using 'tensorboard --logdir {file_path}'.")

    train_writer = SummaryWriter(os.path.join(file_path, 'train'))
    eval_writer = SummaryWriter(os.path.join(file_path, 'test'))

    train_step = make_train_step(config.loss_fn)
    eval_step = make_eval_step(config.loss_fn)
    try:
        training_data_iterator = iter(config.data_config.train_data_loader)
        eval_data_iterator = iter(config.data_config.eval_data_loader)
        with ocp.training.Checkpointer(checkpoint_path) as checkpointer:
            for step in range(1, config.training_steps + 1):
                example = next(training_data_iterator)
                loss = train_step(config.model, config.optimizer, example.data,
                                example.labels)
                train_writer.add_scalar('Training Loss', loss.item(), step)
                print(f'Step {step}: Training loss - {loss.item():.4f}')

                if step % 10 == 0:
                    train_writer.flush()
                    eval_writer.flush()

                if step % config.eval_frequency == 0:
                    example = next(eval_data_iterator)
                    loss = eval_step(config.model, example.data, example.labels)
                    eval_writer.add_scalar('Eval loss', loss.item(), step)
                    print(f'Step {step}: Eval loss - {loss.item():.4f}')

                if step % config.checkpoint_frequency == 0:
                    checkpointer.save(step, config.model)

    finally:
        train_writer.close()
        eval_writer.close()
