"""
Save wav2vec2-base embeddings to an npz file.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.data.utils import construct_dataset
from src.utils import DATA_DIR, MODEL_DIR, PathList, logger, PickleStream, get_best_cuda


def main(
    *,
    audio_path: list[Path],
    model_path: Path,
    output_path: Path,
    device: str = "cuda",
):
    logger.info("USING DEVICE: %s", device)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    pathlist = PathList(audio_path)

    model = Wav2Vec2Model.from_pretrained(model_path)
    model.to(device)
    model.eval()
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    dataset, collator = construct_dataset(pathlist, processor)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
    )
    logger.info("Running inference on %d files", len(dataset))
    logger.info("Using batch size of 32")
    with PickleStream.writer(output_path) as pickler:
        for batch in tqdm(dataloader):
            with torch.no_grad():
                x, mask = batch["input_values"], batch["emission_mask"]
                output = model(x.to(device), output_hidden_states=True).hidden_states
                # apply global pool over time dimension
                output = [
                    (
                        torch.einsum("btd,bt->bd", x.detach().cpu(), mask)
                        / torch.einsum("bt->b", mask).unsqueeze(-1)
                    ).numpy()
                    for x in output
                ]
                reshape = list(zip(*output))
            for r in reshape:
                pickler.dump(r)
    filelist = output_path.parent / f"{output_path.stem}_filelist.txt"
    with open(filelist, "w") as f:
        f.write("\n".join(map(str, pathlist)))
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_path",
        nargs="+",
        type=Path,
        help="Path to audio file(s) or their immediate parent directories.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="The model to use for inference",
        default=MODEL_DIR / "wav2vec2-large-lv60",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the embeddings",
        default=DATA_DIR / "embeddings.npz",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run inference on",
        default=f"cuda:{get_best_cuda()}",
    )

    args = parser.parse_args()
    main(**vars(args))
