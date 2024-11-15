"""
Training script for all the models.
"""

from jsonargparse import lazy_instance
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

from src.models.w2v2_utterance_embedding import UtteranceSERModel  # noqa
from src.models.w2v2_contextual_embedding import ContextualSERModel  # noqa
from src.data.datamodules import MeldEmbeddingsDataModule, ContextualMeldEmbeddingsDataModule  # noqa


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.num_labels",
            "model.init_args.num_labels",
            apply_on="instantiate",
        )


def main():
    MyCLI(
        save_config_callback=None,
        trainer_defaults={
            # Use lazy_instance to defer the instantiation of the logger
            # https://lightning.ai/docs/pytorch/latest/cli/lightning_cli_expert.html#class-type-defaults
            "logger": lazy_instance(WandbLogger, project="speech-emotion"),
            # These callbacks are not tracked in the config files; they will be
            # added to the trainer when this script is run.
            "callbacks": [
                ModelCheckpoint(monitor="val/weighted_f1", mode="max"),
                EarlyStopping(
                    monitor="val/weighted_f1", mode="max", patience=10, min_delta=0.001
                ),
            ],
        },
    )


if __name__ == "__main__":
    main()
