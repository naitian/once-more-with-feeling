"""
Run ffsubsync on the videos and associated subtitles for the movies in the
metadata file that have an associated SRT file
"""

import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm
from ffsubsync.ffsubsync import run, make_parser

from src.utils import DATA_DIR, logger


def main(args):
    args.output_dir.mkdir(exist_ok=True, parents=True)
    metadata = pl.read_csv(args.metadata, has_header=True, separator="\t")
    ffsubsync_parser = make_parser()
    for row in tqdm(metadata.to_pandas().itertuples(), total=len(metadata)):
        imdb = row.imdb
        movie_path = Path(row.path)
        srt_path = args.srt_path / f"{imdb}.srt"
        output_path = args.output_dir / f"{imdb}.srt"

        if not srt_path.exists():
            logger.info(f"Skipping {imdb} because {srt_path} does not exist")
            continue

        if not args.overwrite and output_path.exists():
            logger.info(f"Skipping {imdb} because {output_path} already exists")
            continue

        logger.info(f"Running ffsubsync on {movie_path}")
        result = run(ffsubsync_parser.parse_args([str(movie_path), "-i", str(srt_path), "-o", str(output_path)]))
        if not result["sync_was_successful"]:
            logger.warning(f"Failed to sync {movie_path} ({imdb}) with ffsubsync")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to the metadata file with the movie paths and SRT paths",
    )
    parser.add_argument(
        "--srt_path",
        type=Path,
        help="Path to the directory containing the SRT files",
        default=DATA_DIR / "srt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the directory to save the synced movies",
        default=DATA_DIR / "aligned_srt",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    main(args)
