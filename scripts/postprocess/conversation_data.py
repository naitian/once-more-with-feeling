"""
Generate clusters for contextual inference
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import DATA_DIR, logger
from src.video.meta import VideoMetadata


def main(args):
    output_data_dir = args.output_dir

    for dir in [output_data_dir]:
        dir.mkdir(exist_ok=True, parents=True)

    video = VideoMetadata.from_imdb(args.imdb_id)
    transcript = pd.read_csv(args.asr_dir / f"{video.imdb}.tsv", delimiter="\t")

    transcript.loc[:, "group"] = (
        (transcript.start.shift(-1) - transcript.end)
        .gt(args.min_gap)
        .cumsum()
        .shift()
        .fillna(0)
        .astype(int)
    )
    logger.info(f"Found {transcript.group.nunique()} clusters in {video.imdb}")

    chunks = transcript.groupby("group").agg({"start": "first", "end": "last"})
    chunks = chunks.sample(frac=1)
    runtimes = (chunks.end - chunks.start) / 60

    logger.info(f"Average chunk length: {runtimes.mean()} minutes.")
    logger.info(f"Maximum chunk length: {runtimes.max()} minutes.")
    logger.info(f"Minimum chunk length: {runtimes.min()} minutes.")

    def get_cluster_utterances(group):
        df = transcript.loc[transcript.group == group].copy()
        offset = df.start.min()
        df.start -= offset
        df.end -= offset
        return df[["start", "end", "text", "wav_path"]].to_dict(orient="records")

    output_data = []
    for i, chunk in tqdm(chunks.iterrows(), total=len(chunks)):
        if transcript.group.eq(i).sum() < args.min_turns:
            # Skip clusters with fewer than min_turns
            continue
        start = chunk.start
        end = chunk.end
        cluster = get_cluster_utterances(i)
        chunk_data = {
            "start": start,
            "end": end,
            "duration": end - start,
            "imdb": video.imdb,
            "title": video.title,
            "utterances": [
                {
                    "start": utterance["start"],
                    "end": utterance["end"],
                    "text": utterance["text"],
                }
                for utterance in cluster
            ],
            "audio_clips": [str(utterance["wav_path"]) for utterance in cluster],
        }
        output_data.append(chunk_data)
    with open(output_data_dir / f"{video.imdb}.ndjson", "w") as f:
        f.write("\n".join([json.dumps(chunk) for chunk in output_data]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "imdb_id",
        type=str,
        help="The IMDb ID of the movie to generate annotation data for.",
    )
    parser.add_argument(
        "--asr_dir",
        type=Path,
        help="The directory containing the ASR data.",
        default=DATA_DIR / "asr_extracts/split_data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The directory to save the new data files.",
        default=DATA_DIR / "convo_data",
    )
    parser.add_argument(
        "--min_turns",
        type=int,
        help="The minimum number of turns in a cluster.",
        default=1,
    )
    parser.add_argument(
        "--min_gap",
        type=int,
        help="The minimum gap between clusters in seconds.",
        default=3,
    )
    args = parser.parse_args()
    main(args)
