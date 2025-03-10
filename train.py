""" Train a movie recommendation model using Neural Collaborative Filtering. """

import argparse
import importlib

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.log import logger

DEFAULT_DATA_CLASS = "RatingsDataModule"
DEFAULT_LIT_MODEL_CLASS = "LitNeuralCollaborativeFilteringModel"
DEFAULT_MODEL_CLASS = "NeuralCollaborativeFilteringModel"
DEFAULT_EARLY_STOPPING = 10

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Faster, but less precise than default value "highest"
torch.set_float32_matmul_precision("high")


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add pl.Trainer args to use
    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_group.add_argument(
        "--accelerator", default="auto", help="Lightning Trainer accelerator"
    )
    trainer_group.add_argument("--devices", default="auto", help="Number of GPUs")
    trainer_group.add_argument("--max_epochs", type=int, default=-1)
    trainer_group.add_argument("--fast_dev_run", type=int, default=0)
    trainer_group.add_argument("--overfit_batches", type=float, default=0.0)

    # Basic arguments
    parser.add_argument("--data_class", type=str, default=DEFAULT_DATA_CLASS)
    parser.add_argument("--lit_model_class", type=str, default=DEFAULT_LIT_MODEL_CLASS)
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS)
    parser.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"retrieval_model_training.data.{temp_args.data_class}")
    lit_model_class = _import_class(
        f"retrieval_model_training.lit_models.{temp_args.lit_model_class}"
    )
    model_class = _import_class(
        f"retrieval_model_training.models.{temp_args.model_class}"
    )

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)
    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model_class.add_to_argparse(lit_model_group)
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py \
        --max_epochs=10 \
        --devices=0 \
        --num_workers=20
        --model_class=NeuralCollaborativeFilteringModel \
        --data_class=RatingsDataModule
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"retrieval_model_training.data.{args.data_class}")
    lit_model_class = _import_class(
        f"retrieval_model_training.lit_models.{args.lit_model_class}"
    )
    model_class = _import_class(f"retrieval_model_training.models.{args.model_class}")

    data = data_class(args=vars(args))

    if args.model_class == "NeuralCollaborativeFilteringModel":
        # NeuralCollaborativeFilteringModel needs the number of users and movies
        # only available after the data is prepared
        data.prepare_data()
        data.setup()
        args.num_users = data.num_user_labels()
        args.num_movies = data.num_movie_labels()
        model = model_class(
            data.num_user_labels(), data.num_movie_labels(), args=vars(args)
        )
    else:
        model = model_class(args=vars(args))

    lit_model = lit_model_class(model, args=vars(args))

    if args.overfit_batches:
        if args.overfit_batches.is_integer():
            args.overfit_batches = int(args.overfit_batches)
        # There's no available val_loss when overfitting to batches
        loss_to_log = "train_loss"
        enable_checkpointing = False
    else:
        loss_to_log = "val_loss"
        enable_checkpointing = True

    early_stopping_callback = EarlyStopping(
        monitor=loss_to_log, mode="min", patience=args.early_stopping
    )

    model_checkpoint_callback = ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.2f}",
        monitor=loss_to_log,
        mode="min",
    )
    callbacks = (
        [early_stopping_callback]
        if args.overfit_batches
        else [early_stopping_callback, model_checkpoint_callback]
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        callbacks=callbacks,
        logger=TensorBoardLogger("training/logs"),
        enable_checkpointing=enable_checkpointing,
    )
    trainer.fit(lit_model, datamodule=data)
    if not args.overfit_batches:
        trainer.test(lit_model, datamodule=data)
        trainer.predict(lit_model, datamodule=data)

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        logger.info("Best model saved at %s:", best_model_path)

    return trainer


if __name__ == "__main__":
    main()
