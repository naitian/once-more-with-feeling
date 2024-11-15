import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.datamodules import EmbeddingsDataset, MeldEmbeddingsDataModule
from src.models.w2v2_utterance_embedding import UtteranceSERModel
from src.utils import DATA_DIR, MODEL_DIR, CsvWriter, get_best_cuda


class InferenceDataset(EmbeddingsDataset):
    def __init__(self, stats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_standardization(stats)


def collator(batch):
    embeddings = torch.stack([x["embeddings"] for x in batch])
    paths = [x["path"] for x in batch]
    return {"embeddings": embeddings, "paths": paths}


def main(args):
    stats = pickle.load((MODEL_DIR / "ser/stats.pkl").open("rb"))
    dataset = InferenceDataset(
        embeddings_path=args.embeddings_file,
        stats=stats,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
    )
    model = UtteranceSERModel.load_from_checkpoint(
        MODEL_DIR / "ser/utterance_stats.ckpt"
    )
    device = f"cuda:{get_best_cuda()}"
    model.to(device)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.output_dir / f"{args.embeddings_file.stem}.tsv"
    with CsvWriter.writer(output_file, delimiter="\t") as writer:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                embeddings = batch["embeddings"]
                paths = batch["paths"]
                embeddings = embeddings.to(device)
                batch_logits = F.softmax(model(embeddings), dim=1).to("cpu").numpy()
                for path, logits in zip(paths, batch_logits):
                    writer.dump(
                        {
                            "path": path,
                            **{
                                MeldEmbeddingsDataModule.idx_to_label[i]: logit
                                for i, logit in enumerate(logits)
                            },
                        }
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "embeddings_file",
        type=Path,
        help="The directory containing the data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size to use for training.",
        default=32,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The dir to save the inference results.",
        default=DATA_DIR / "ser_inference",
    )
    args = parser.parse_args()
    main(args)
