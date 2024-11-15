""" "
Runs inference on a given model and list of audio files.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from tqdm import tqdm

from src.models import Wav2Vec2ConformerSERModel
from src.utils import DATA_DIR, PathList, logger
from src.data.utils import construct_dataset


# TODO: move this to src/utils.py
# potentially integrate with PathList.
def get_paths(filepath):
    """
    Get the paths from a file containing paths
    """
    with open(filepath, "r") as f:
        return [Path(x.strip()) for x in f.readlines()]


def main(args):
    pathlist = PathList(args.audio_files)
    pathlist = [
        y for x in pathlist for y in ([x] if x.suffix == ".wav" else get_paths(x))
    ]
    logger.info("Running inference on %d files", len(pathlist))

    model = Wav2Vec2ConformerSERModel.load_from_checkpoint(args.model)
    device = model.device
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model.hparams.base_model_name
    )

    dataset, collator = construct_dataset(pathlist, feature_extractor)
    outputs = []
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
    )
    for batch in tqdm(dataloader):
        with torch.no_grad():
            x, mask = batch["input_values"], batch["emission_mask"]
            output = model(x.to(device), mask.to(device)).detach().cpu().numpy()
        outputs.append(output)
    all_outputs = np.concatenate(outputs, axis=0)
    output_path = args.output_dir / "inference_output.npy"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    logger.info("Writing results")
    with open(output_path, "wb") as f:
        np.save(f, all_outputs)
    with open(args.output_dir / "audio_paths.txt", "w") as f:
        f.write("\n".join(map(str, pathlist)))
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="The model to use for inference")
    parser.add_argument(
        "audio_files", nargs="+", help="The audio files to run inference on"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The directory to save the output to",
        default=DATA_DIR / "inference_output",
    )
    args = parser.parse_args()
    main(args)
