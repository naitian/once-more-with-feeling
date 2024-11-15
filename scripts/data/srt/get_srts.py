"""Copy the relevant SRT files from the SRT folder to the data folder."""

import argparse
from datetime import timedelta
from pathlib import Path

import polars as pl
import srt
from tqdm import tqdm

from src.utils import DATA_DIR, LOG_DIR, logger

SRT_DIR = Path("/mnt/data0/corpora/opensubtitles/all_subtitles")


def _parse_spreadsheet_time_fmt(time_str):
    """
    Parse the time format in the spreadsheet of H:MM:SS
    """
    return timedelta(
        hours=int(time_str[0]), minutes=int(time_str[2:4]), seconds=int(time_str[5:7])
    )


def copy_and_fix_srt(imdb, start, output_dir):
    """
    srt_path is the path to the SRT file
    start is in H:MM:SS format

    overwrite the SRT file with the start time
    """
    srt_path = SRT_DIR / f"{imdb}.srt"
    if start is None:
        logger.info(f"No start time for SRT file {srt_path}")
        offset = timedelta(0)
    else:
        logger.info(f"Fixing SRT file {srt_path} with start time {start}")
        offset = _parse_spreadsheet_time_fmt(start)
    try:
        srt_generator = srt.parse(open(srt_path).read())
        new_entries = []
        for entry in srt_generator:
            new_entries.append(
                srt.Subtitle(
                    index=entry.index,
                    start=entry.start + offset,
                    end=entry.end + offset,
                    content=entry.content,
                )
            )
    except Exception as e:
        logger.error(f"Error parsing SRT {srt_path}: {e}")
        return

    with open(output_dir / srt_path.name, "w") as f:
        f.write(srt.compose(new_entries))


def main(args):
    logger.info(f"Copying SRT files for movies in {args.metadata}")
    args.output_dir.mkdir(exist_ok=True, parents=True)
    df = pl.read_csv(args.metadata, has_header=True, separator="\t")
    for x in tqdm(df.select(pl.col("imdb", "start")).iter_rows()):
        copy_and_fix_srt(*x, args.output_dir)


if __name__ == "__main__":
    logfile = LOG_DIR / "runs/get_srts.log"
    logfile.parent.mkdir(exist_ok=True, parents=True)

    parser = argparse.ArgumentParser(
        description="Copy the relevant SRT files from the SRT folder to the data folder."
    )
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to the metadata file of movies we're interested in. This file should be a TSV with the `imdb` column.",
    )
    parser.add_argument(
        "--srt_dir",
        type=Path,
        help="Path to the directory containing the SRT files.",
        default=SRT_DIR,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the directory to copy the SRT files to.",
        default=DATA_DIR / "srt",
    )
    args = parser.parse_args()
    main(args)
