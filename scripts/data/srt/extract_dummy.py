"""
Takes the same inputs and outputs as extract_audio.py, but doesn't extract the
audio. It just creates the directory structure and the text data files.

Output a directory of WAV files, one for each snippet of text
Also output a CSV file with the following columns:
- start
- end
- wav_path
- text
"""

import argparse
from pathlib import Path


import polars as pl
import srt
from tqdm import tqdm

from src.utils import DATA_DIR, logger


def extract_audio(imdb, movie_path, srt_dir, output_dir):
    logger.info(f"Extracting audio for {imdb}")
    aligned_srt = srt_dir / f"{imdb}.srt"
    if not aligned_srt.exists():
        logger.warning(f"Skipping... No aligned SRT for {imdb}")
        return

    audio_dir = output_dir / "audio" / imdb
    meta_path = output_dir / "data" / f"{imdb}.csv"

    audio_dir.mkdir(exist_ok=True, parents=True)
    meta_path.parent.mkdir(exist_ok=True, parents=True)

    result = []
    # Load the SRT file
    with open(aligned_srt) as f:
        srt_file = srt.parse(f.read())
        for i, subtitle in enumerate(tqdm(list(srt_file))):
            start = subtitle.start.total_seconds()
            end = subtitle.end.total_seconds()
            text = subtitle.content

            # Extract the audio
            result.append(
                {
                    "start": start,
                    "end": end,
                    "wav_path": None,
                    "text": text,
                }
            )
    meta_df = pl.DataFrame(result)
    meta_df.write_csv(meta_path, separator="\t")


def main(args):
    df = pl.read_csv(args.metadata, has_header=True, separator="\t")
    for imdb_id, movie_path in df.select(pl.col("imdb", "path")).rows():
        extract_audio(imdb_id, Path(movie_path), args.srt_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to the metadata file with the movie paths",
    )
    parser.add_argument(
        "--srt_dir",
        type=Path,
        help="Path to the directory with the cleaned, aligned SRT files",
        default=DATA_DIR / "cleaned_srt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the directory to save the audio files",
        default=DATA_DIR / "dummy_extracts",
    )
    args = parser.parse_args()
    main(args)
