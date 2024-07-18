""" Train a movie recommendation model. """

import argparse
import importlib

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import lit_models
from src.utils.log import logger

DEFAULT_DATA_CLASS = "MovieLensDataModule"
DEFAULT_MODEL_CLASS = "RecommendationModel"
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
    trainer_group.add_argument("--fast_dev_run", type=bool, default=False)
    trainer_group.add_argument("--overfit_batches", type=float, default=0.0)

    # Basic arguments
    parser.add_argument("--data_class", type=str, default=DEFAULT_DATA_CLASS)
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS)
    parser.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"src.data.{temp_args.data_class}")
    model_class = _import_class(f"src.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    # NOTE: Hardcoded for now, but can be made dynamic
    lit_models.LitRecommender.add_to_argparse(lit_model_group)

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
        --model_class=RecommendationModel \
        --data_class=MovieLensDataModule
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"src.data.{args.data_class}")
    model_class = _import_class(f"src.models.{args.model_class}")

    data = data_class(args=vars(args))
    # Prepare data so that we can get the config for the model
    data.prepare_data()
    data.setup()

    model = model_class(
        num_users=data.num_user_labels(),
        num_movies=data.num_movie_labels(),
        args=vars(args),
    )
    # TODO: Add args=vars(args) to LitRecommender when it's implemented
    lit_model = lit_models.LitRecommender(model)

    tb_logger = TensorBoardLogger("training/logs")

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
        callbacks=callbacks,  # type: ignore
        logger=tb_logger,
        enable_checkpointing=enable_checkpointing,
    )
    trainer.fit(lit_model, datamodule=data)
    # TODO: Uncomment this line when LitRecommender test_step() is implemented
    trainer.test(lit_model, datamodule=data)

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        logger.info("Best model saved at %s:", best_model_path)


if __name__ == "__main__":
    main()
