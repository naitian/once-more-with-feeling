"""
Generate clusters and movies for annotation
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from src.utils import DATA_DIR, logger
from src.video.meta import VideoMetadata


def main(args):
    output_video_dir = args.output_dir / "videos"
    output_data_dir = args.output_dir / "data"

    for dir in [output_video_dir, output_data_dir]:
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
        return df[["start", "end", "text"]].to_dict(orient="records")

    output_data = []
    video_clip = VideoFileClip(str(video.path))
    for i, chunk in tqdm(chunks.iterrows(), total=len(chunks)):
        if len(output_data) >= args.num_samples:
            # Stop after num_samples
            break
        if transcript.group.eq(i).sum() < args.min_turns:
            # Skip clusters with fewer than min_turns
            continue
        if (chunk.end - chunk.start) > (2 * 60):
            # Skip clusters longer than 2 minutes
            continue
        start = chunk.start
        end = chunk.end
        clip_name = f"{video.imdb}_{i}.mp4"
        chunk_data = {
            "start": start,
            "end": end,
            "duration": end - start,
            "imdb": video.imdb,
            "title": video.title,
            "utterances": get_cluster_utterances(i),
            "clip_name": clip_name,
        }
        output_data.append(chunk_data)
        clip = video_clip.subclip(start, end).resize(height=360)
        clip.write_videofile(
            str(output_video_dir / clip_name),
            codec="libx264",
            audio_codec="aac",
        )
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
        default=DATA_DIR / "annotation_data",
    )
    parser.add_argument(
        "--min_turns",
        type=int,
        help="The minimum number of turns in a cluster.",
        default=2,
    )
    parser.add_argument(
        "--min_gap",
        type=int,
        help="The minimum gap between clusters in seconds.",
        default=3,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="The number of samples to generate.",
        default=10,
    )
    args = parser.parse_args()
    main(args)
