import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import Accuracy
from transformers import Wav2Vec2ConformerModel


class Wav2Vec2ConformerSERModel(L.LightningModule):
    LOCAL = "local"
    CONTEXTUAL = "contextual"
    ALL = "all"

    def __init__(
        self,
        base_model_name="facebook/wav2vec2-conformer-rope-large",
        model_type="contextual",
        speaker_norm=False,
        num_labels=7,
        num_layers=1,
        stats_path=None,
        learning_rate=1e-4,
    ):
        super().__init__()
        if model_type not in [self.LOCAL, self.CONTEXTUAL, self.ALL]:
            raise ValueError(f"Invalid model type: {self.hparams.model_type}")

        self.stats = None

        # freeze the weights of the wav2vec model
        self.embedding = Wav2Vec2ConformerModel.from_pretrained(base_model_name)
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.embedding_layers = 25
        self.embedding_dim = 1024

        self.feature_fc = nn.Linear(512, 128)
        self.activation_fc = nn.Linear(self.embedding_dim, 128)
        self.activation_wts = nn.Parameter(torch.ones(self.embedding_layers))

        self.mlp = nn.ModuleList([nn.Linear(128, 128) for _ in range(num_layers)])
        self.output = nn.Linear(128, num_labels)
        self.dropout = nn.Dropout(0.2)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_labels)
        self.save_hyperparameters()

    def forward(self, x, mask, embedding=None):
        if embedding is None:
            x = self.embedding(x, output_hidden_states=True)
        else:
            x = embedding

        if self.hparams.speaker_norm:
            raise NotImplementedError
        elif self.stats is None:
            local_state = self.feature_fc(x.extract_features)
            output_state = x.hidden_states
        else:
            # local state is not normalized
            local_state = self.feature_fc(x.extract_features)
            output_state = tuple(
                (layer - self.stats[i]["mean"].to(self.device))
                / self.stats[i]["std"].to(self.device)
                for i, layer in enumerate(x.hidden_states)
            )

        if self.hparams.model_type == self.LOCAL:
            x = self.feature_fc(local_state)
        elif self.hparams.model_type == self.CONTEXTUAL:
            x = self.activation_fc(output_state[-1])
        else:
            activations = torch.stack(output_state, dim=1)
            x = torch.einsum("bltd,l->btd", activations, self.activation_wts)
            x = self.activation_fc(x)

        x = F.relu(x)
        for layer in self.mlp:
            x = F.relu(layer(self.dropout(x)))

        x = x * mask.unsqueeze(-1)
        x = x.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        x = self.output(x)
        return x  # return the logits, not the softmax

    def training_step(self, batch, batch_idx):
        x, y, mask = batch.input_values, batch.labels, batch.emission_mask
        y_pred = self(x.to(self.device), mask.to(self.device))

        loss = F.cross_entropy(y_pred, y.to(self.device))
        self.train_accuracy(y_pred, y.to(self.device))

        self.log_dict(
            {"train/loss": loss, "train/accuracy": self.train_accuracy},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch.input_values, batch.labels, batch.emission_mask
        y_pred = self(x.to(self.device), mask.to(self.device))

        loss = F.cross_entropy(y_pred, y.to(self.device))
        self.val_accuracy(y_pred, y.to(self.device))

        self.log_dict(
            {"val/loss": loss, "val/accuracy": self.val_accuracy},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
