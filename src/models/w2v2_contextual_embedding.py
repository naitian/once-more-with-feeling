"""
This model takes sequence of model activations as input and outputs an emotion
prediction label.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, BootStrapper, F1Score, MetricCollection


def apply_packed(packed_sequence, fn):
    """Apply a function to a packed sequence."""
    unpacked, lengths = nn.utils.rnn.pad_packed_sequence(packed_sequence)
    return nn.utils.rnn.pack_padded_sequence(
        fn(unpacked), lengths, enforce_sorted=False
    )


class ContextualSERModel(L.LightningModule):
    def __init__(
        self,
        num_labels: int = 7,
        num_layers: int = 1,
        num_lstm_layers: int = 1,
        activation_shapes: list[tuple[int]] = [(1024,)] * 25,
        dropout: float = 0.2,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        if len(set(activation_shapes)) != 1:
            raise NotImplementedError(
                "Different activation shapes at different layers not supported."
            )

        # in this version of the model, we pool the activation layers before
        # feeding through a biLSTM, then feedforward MLP
        self.activation_wts = nn.Parameter(torch.ones(len(activation_shapes)))
        self.activation_fc = nn.Linear(activation_shapes[0][0], hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout,
        )

        self.output = nn.Linear(2 * hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)  # TODO: configurable dropout

        metrics = MetricCollection(
            {
                "accuracy": BootStrapper(Accuracy(task="multiclass", num_classes=num_labels)),
                "f1": BootStrapper(F1Score(task="multiclass", num_classes=num_labels)),
                "weighted_f1": BootStrapper(F1Score(
                    task="multiclass", num_classes=num_labels, average="weighted"
                )),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.save_hyperparameters()

    def forward(self, x):
        # x: (batch, num_activations, activation_dim)
        x = apply_packed(
            x, lambda x: torch.einsum("tbld,l->tbd", x, self.activation_wts)
        )
        x = apply_packed(x, self.activation_fc)
        x, _ = self.lstm(x)
        x = apply_packed(x, self.dropout)
        x = apply_packed(x, self.output)
        x = torch.concat(nn.utils.rnn.unpack_sequence(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch["embeddings"], batch["labels"]
        y_pred = self(x.to(self.device))

        loss = F.cross_entropy(y_pred, y.to(self.device))

        self.train_metrics(y_pred, y.to(self.device))

        self.log_dict(
            {"train/loss": loss, **self.train_metrics.compute()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(y),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["embeddings"], batch["labels"]
        y_pred = self(x.to(self.device))

        loss = F.cross_entropy(y_pred, y.to(self.device))
        self.val_metrics(y_pred, y.to(self.device))

        self.log_dict(
            {"val/loss": loss, **self.val_metrics.compute()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["embeddings"], batch["labels"]
        y_pred = self(x.to(self.device))

        loss = F.cross_entropy(y_pred, y.to(self.device))
        self.test_metrics(y_pred, y.to(self.device))

        self.log_dict(
            {"test/loss": loss, **self.test_metrics.compute()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
