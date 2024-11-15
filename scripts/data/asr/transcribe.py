"""
Use faster-whisper to transcribe audio files.
"""

import argparse

import polars as pl
from faster_whisper import WhisperModel
from tqdm import tqdm

from src.utils import DATA_DIR, MODEL_DIR, PathList, logger, get_best_cuda
from src.video.meta import VideoMetadata

# model = WhisperModel("large-v3", device="cuda", compute_type="int8_float32")
model = WhisperModel(
    str(MODEL_DIR / "distil-medium.en"),
    device="cuda",
    device_index=get_best_cuda(),
    compute_type="int8_float32",
)


def transcribe_wav(path):
    """
    path: Path to the wav file
    """
    try:
        segments, _ = model.transcribe(path, language="en")
        segments = list(segments)
    except Exception as e:
        logger.error(f"Error transcribing {path}: {e}")
        return
    for segment in segments:
        if (
            segment.avg_logprob < -0.693 or segment.no_speech_prob > 0.8
        ):  # -0.693 is the log(0.5)
            logger.warning(f"Low confidence detected for {path} -- {segment.text}")
            return
    return " ".join([x.text.strip() for x in segments])


def transcribe(path, overwrite=False, ignorelist=None):
    """
    path: path to the directory containing all the wav files for a specific movie
    """
    if not path.exists():
        logger.warning(f"{path} does not exist. Skipping")
        return
    imdb_id = path.parts[-1]
    data_file = path.parents[1] / "data" / f"{imdb_id}.tsv"
    logger.info(f"Transcribing {imdb_id}")

    ignorelist = ignorelist or []
    if imdb_id in ignorelist:
        logger.info(f"{imdb_id} is in the ignore list. Skipping")
        return

    try:
        df = pl.read_csv(data_file, has_header=True, separator="\t")
    except pl.exceptions.NoDataError:
        logger.warning(f"{data_file} was empty. Skipping {imdb_id}")
        return
    if not overwrite and df["text"].null_count() == 0:
        logger.info(f"{imdb_id} already transcribed. Skipping")
        return
    new_results = []
    for row in tqdm(df.iter_rows(named=True)):
        text = transcribe_wav(row["wav_path"])
        row["text"] = text
        new_results.append(row)

    new_df = pl.DataFrame(new_results)
    new_df.write_csv(data_file, separator="\t")
    logger.info("Done")


def main(args):
    if args.all is None and args.extract_path is None:
        raise ValueError("Please provide either --all or --extract_path")
    if args.all:
        pathlist = [
            DATA_DIR / "asr_extracts" / "audio" / x.imdb
            for x in VideoMetadata.itermovies()
        ]
        pathlist = [x for x in pathlist if x.exists()]
    else:
        pathlist = PathList(args.extract_path, expand_dirs=False)

    if args.ignore:
        logger.info(f"Ignoring {args.ignore}")
    for path in tqdm(pathlist):
        transcribe(path, overwrite=args.overwrite, ignorelist=args.ignore)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_path", nargs="+", help="Path to audio file(s)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Transcribe all the movies in the metadata file",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing transcriptions"
    )
    parser.add_argument("--ignore", nargs="+", help="Ignore these movies")
    args = parser.parse_args()
    main(args)
