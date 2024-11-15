"""
Pytorch Lightning DataModules for all of the datasets
"""

import math
import pickle
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from src.utils import DATA_DIR, PickleStream


def create_collator(feature_extractor, max_length):
    def collator(examples):
        def _calculate_offset(size):
            # calculate the index in the emission given the waveform
            # from here: https://github.com/pytorch/audio/blob/main/examples/self_supervised_learning/data_modules/_utils.py#L360
            kernel_size = 25
            stride = 20
            sample_rate = 16  # 16 per millisecond
            return max(
                math.floor((size - kernel_size * sample_rate) / (stride * sample_rate))
                + 1,
                0,
            )

        batch = pad_without_fast_tokenizer_warning(
            feature_extractor,
            examples,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        max_emission_length = _calculate_offset(batch["input_values"].shape[-1])
        sizes = [_calculate_offset(len(x["input_values"])) for x in examples]
        emission_mask = torch.stack(
            [F.pad(torch.ones(size), (0, max_emission_length - size)) for size in sizes]
        )
        batch["emission_mask"] = emission_mask
        return batch

    return collator


def create_processor(feature_extractor):
    def process(examples):
        audios = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audios, truncation=True, sampling_rate=16_000, max_length=128_000
        )
        return inputs

    return process


class EmbeddingsDataset:
    """
    Dataset class for embeddings.
    """

    def __init__(self, embeddings_path: Path):
        self.data = self.load_data(embeddings_path)

    def load_data(self, embeddings_path: Path) -> None:
        data = []
        pathlist_path = embeddings_path.parent / f"{embeddings_path.stem}_filelist.txt"
        for path, embeddings in zip(
            open(pathlist_path, "r"),
            PickleStream.read(embeddings_path),
        ):
            embeddings = np.stack(embeddings, axis=0)
            data.append({"path": path.strip(), "embeddings": torch.tensor(embeddings)})
        return data

    def apply_standardization(self, stats):
        for i, _ in enumerate(self.data):
            self.data[i]["embeddings"] = torch.stack(
                [
                    (x - stat["mean"]) / stat["std"]
                    for x, stat in zip(self.data[i]["embeddings"], stats)
                ],
                dim=0,
            )

    def add_metadata(self, metadata):
        for i, _ in enumerate(self.data):
            self.data[i].update(metadata[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ContextualEmbeddingsDataset(EmbeddingsDataset):
    def __init__(self, embeddings_path: Path, conversation_groups: list[int] = None):
        """
        conversation_groups: list of ints, where each int represents the conversation ID of the corresponding embedding
        """
        super().__init__(embeddings_path)
        if conversation_groups is not None:
            self.load_conversation_groups(conversation_groups)

    def load_conversation_groups(self, conversation_groups: list[int]):
        conversation_ids = set(conversation_groups)
        self.conversations = [
            # TODO: this is not efficient, since it loops thru the entire dataset for each conversation
            # but it's fine for now since the dataset is small
            # consider optimizing this if the dataset grows (e.g. potentially when running inference)
            [i for i, x in enumerate(conversation_groups) if x == conversation_id]
            for conversation_id in conversation_ids
        ]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = [self.data[i] for i in self.conversations[idx]]
        metadata = {
            meta_key: [x[meta_key] for x in conversation]
            for meta_key in conversation[0]
            if meta_key != "embeddings"
        }
        if "label" in metadata:
            metadata["labels"] = torch.tensor(metadata["label"])
            del metadata["label"]

        return {
            "embeddings": torch.stack([x["embeddings"] for x in conversation]),
            **metadata,
        }


class BaseEmbeddingsDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_batch_size=32,
        val_batch_size=32,
        standardize: bool = False,
        stats_path: Path | None = None,
        dataset_class: type[EmbeddingsDataset] = EmbeddingsDataset,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_class = dataset_class

        self.standardize = standardize
        self.stats_path = stats_path
        self.datasets = {}

    def calculate_and_save_stats(self, embeddings, stats_path):
        stats = [
            {"mean": torch.tensor(y.mean(0)), "std": torch.tensor(y.std(0))}
            for x in zip(*embeddings)
            if (y := np.stack(x, 0))
            is not None  # just for the assignment expression; no need for the None check otherwise
        ]
        pickle.dump(stats, open(stats_path, "wb"))
        return stats

    def load_stats(self, stats_path):
        return pickle.load(open(stats_path, "rb"))

    def apply_standardization(self):
        for split in ["train", "dev", "test"]:
            self.datasets[split].apply_standardization(self.stats)

    def prepare_data(self):
        # load the embeddings from the disk
        for split in ["train", "dev", "test"]:
            embeddings_path = self.embeddings_dir / split / f"{split}.pkl"
            self.datasets[split] = self.dataset_class(embeddings_path=embeddings_path)

        if self.standardize:
            if self.stats_path is None:
                raise ValueError("Must provide stats path if standardizing")
            if self.stats_path.exists():
                self.stats = self.load_stats(self.stats_path)
            else:
                self.stats = self.calculate_and_save_stats(
                    [x["embeddings"] for x in self.datasets["train"]],
                    self.stats_path,
                )
            self.apply_standardization()


class MeldEmbeddingsDataModule(BaseEmbeddingsDataModule):
    idx_to_label = [
        "neutral",
        "joy",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(
            stats_path=DATA_DIR / "datasets/meld/stats.pkl", *args, **kwargs
        )
        self.num_labels = len(self.idx_to_label)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.idx_to_label)}
        self.save_hyperparameters()

    def add_metadata(
        self, embeddings_dataset: EmbeddingsDataset, metadata: pd.DataFrame
    ):
        """
        metadata is a dataframe
        embeddings_dataset is the EmbeddingsDataset object
        """
        metadata.loc[:, "path"] = metadata.wav_path.apply(lambda x: Path(x).stem)
        metadata = metadata.set_index("path")
        metadata.loc[:, "label"] = metadata["Emotion"].apply(self.label_to_idx.get)
        metadata = metadata[["label"]]
        metadata = metadata.to_dict(orient="index")
        metadata = [metadata[Path(x["path"]).stem] for x in embeddings_dataset.data]
        embeddings_dataset.add_metadata(metadata)

    def collator(self, batch):
        embeddings = torch.stack([x["embeddings"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return {"embeddings": embeddings, "labels": labels}

    def prepare_data(self):
        self.embeddings_dir = DATA_DIR / "datasets" / "meld"
        super().prepare_data()
        self.add_metadata(
            self.datasets["train"],
            pd.read_csv(self.embeddings_dir / "train/train.tsv", sep="\t"),
        )
        self.add_metadata(
            self.datasets["dev"],
            pd.read_csv(self.embeddings_dir / "dev/dev.tsv", sep="\t"),
        )
        self.add_metadata(
            self.datasets["test"],
            pd.read_csv(self.embeddings_dir / "test/test.tsv", sep="\t"),
        )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["dev"],
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.collator,
        )



class ContextualMeldEmbeddingsDataModule(MeldEmbeddingsDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            dataset_class=ContextualEmbeddingsDataset,
            *args,
            **kwargs,
        )

    def add_metadata(
        self, embeddings_dataset: ContextualEmbeddingsDataset, metadata: pd.DataFrame
    ):
        metadata = metadata[metadata.wav_path.apply(lambda x: Path(x).exists())]
        super().add_metadata(embeddings_dataset, metadata)
        embeddings_dataset.load_conversation_groups(metadata.group)

    def collator(self, batch):
        embeddings = pack_sequence(
            [x["embeddings"] for x in batch], enforce_sorted=False
        )
        labels = torch.concat([x["labels"] for x in batch])
        return {"embeddings": embeddings, "labels": labels}


if __name__ == "__main__":
    dm = ContextualMeldEmbeddingsDataModule()
    dm.prepare_data()
    dl = dm.train_dataloader()
    for batch in tqdm(dl):
        pass


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_model_name,
        train_batch_size=32,
        val_batch_size=32,
        max_length=128_000,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.encoder_name = base_model_name
        self.max_length = max_length  # 8 seconds of audio at 16kHz

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.encoder_name)
        self.collator = create_collator(self.feature_extractor, self.max_length)

    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage=None):
        process = create_processor(self.feature_extractor)
        self.dataset = self.dataset.map(process, batched=True, remove_columns=["audio"])
        self.dataset = self.dataset.select_columns(["input_values", "label"])
        self.dataset = self.dataset.with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=8,
        )

    def compute_stats(self, model, stats_path):
        # TODO: there must be a more elegant way to do this? without having to
        # do so many of these manual pytorch lighting things?
        self.prepare_data()
        self.setup("fit")
        model.to("cuda")
        dataloader = self.train_dataloader()

        all_embeddings = list(
            torch.zeros((0, model.embedding_dim)) for _ in range(model.embedding_layers)
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                output = model.embedding(
                    batch["input_values"].to("cuda"), output_hidden_states=True
                ).hidden_states
                for i, embedding in enumerate(output):
                    avg_embedding = (
                        torch.einsum(
                            "btd,bt->bd", embedding.cpu(), batch["emission_mask"].cpu()
                        )
                        / torch.einsum("bt->b", batch["emission_mask"].cpu())[:, None]
                    )
                    all_embeddings[i] = torch.cat(
                        [all_embeddings[i], avg_embedding], dim=0
                    )
        stats = tuple({"mean": x.mean(0), "std": x.std(0)} for x in all_embeddings)
        pickle.dump(stats, open(stats_path, "wb"))
        return stats


class IemocapDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_labels = 4
        self.save_hyperparameters()

    def prepare_data(self):
        self.dataset = load_from_disk(DATA_DIR / "iemocap-1234-5-squashed")

    def setup(self, stage: str):
        super().setup(stage)
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["dev"]


class MeldDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_labels = 7
        self.save_hyperparameters()

    def prepare_data(self):
        self.dataset = load_from_disk(DATA_DIR / "meld")
        self.dataset = self.dataset.rename_column("Emotion", "label")

    def setup(self, stage: str):
        super().setup(stage)
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["dev"]
