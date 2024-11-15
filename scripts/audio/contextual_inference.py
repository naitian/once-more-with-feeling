import argparse
import itertools
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from tqdm import tqdm

from src.data.datamodules import ContextualEmbeddingsDataset, MeldEmbeddingsDataModule
from src.models.w2v2_contextual_embedding import ContextualSERModel
from src.utils import DATA_DIR, MODEL_DIR, CsvWriter, get_best_cuda


class ContextualInferenceDataset(ContextualEmbeddingsDataset):
    def __init__(self, stats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_standardization(stats)


def collator(batch):
    embeddings = pack_sequence([x["embeddings"] for x in batch], enforce_sorted=False)
    paths = [x["path"] for x in batch]
    return {"embeddings": embeddings, "paths": paths}


def main(args):
    stats = pickle.load((MODEL_DIR / "ser/stats.pkl").open("rb"))

    dataset = pd.read_json(args.info_file, lines=True)
    fname_to_convo = {}
    for i, clip in dataset.iterrows():
        audio_clips = clip.audio_clips
        for audio_clip in audio_clips:
            fname_to_convo[Path(audio_clip).name] = i
    pathlist_path = (
        args.embeddings_file.parent / f"{args.embeddings_file.stem}_filelist.txt"
    )

    counter = -1
    def get_convo_id(name):
        nonlocal counter
        if name in fname_to_convo:
            return fname_to_convo[name]
        counter -= 1
        return counter
        
    pathlist = pathlist_path.read_text().splitlines()
    convo_groups = [get_convo_id(Path(path).name) for path in pathlist]

    dataset = ContextualInferenceDataset(
        embeddings_path=args.embeddings_file,
        conversation_groups=convo_groups,
        stats=stats,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
    )
    model = ContextualSERModel.load_from_checkpoint(
        MODEL_DIR / "ser/contextual_stats.ckpt"
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
                for path, logits in zip(
                    itertools.chain.from_iterable(paths), batch_logits
                ):
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
        "--info_file",
        type=Path,
        help="The file containing the conversation info.",
        default=DATA_DIR / "annotation_data/data_audio.ndjson",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The dir to save the inference results.",
        default=DATA_DIR / "ser_inference",
    )
    args = parser.parse_args()
    main(args)
