"""Reads all dialogue extract files and deduplicates the text."""

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from src.data.srt_text import load_srt_text
from src.utils import DATA_DIR, PathList


def main(args):
    pathlist = PathList(
        [args.transcript_path / fname for fname in os.listdir(args.transcript_path)]
    )
    srt_texts = load_srt_text(pathlist).to_pandas()

    # deduplicate
    dedup = set()
    for row in tqdm(srt_texts.itertuples(), total=len(srt_texts)):
        dedup.add(row.text)

    # write to file
    outfile = args.output_dir / "text.tsv"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    with open(outfile, "w") as f:
        for i, text in enumerate(dedup):
            f.write(f"{i}\t{text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript_path",
        type=Path,
        default=DATA_DIR / "asr_extracts" / "split_data",
        help="Path to the transcript folder",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_DIR / "text_embeddings",
        help="Output directory",
    )
    main(parser.parse_args())
