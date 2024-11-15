"""
Perform speaker segmentation on videos
"""

import argparse
from pathlib import Path

import polars as pl
import soundfile as sf
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from tqdm import tqdm

from src.utils import DATA_DIR, MODEL_DIR, PathList, logger
from src.video.io import audio_file
from src.video.meta import VideoMetadata


def extract_audio(video_meta, output_dir, pipeline):
    imdb = video_meta.imdb
    video_file = video_meta.path

    audio_dir = output_dir / "audio" / imdb
    meta_path = output_dir / "data" / f"{imdb}.tsv"
    audio_dir.mkdir(exist_ok=True, parents=True)
    meta_path.parent.mkdir(exist_ok=True, parents=True)

    # load the audio from the video
    try:
        audio_path = audio_file(video_file)
    except:  # noqa lol
        logger.error(f"Failed to extract audio for {video_meta}")
        return

    logger.info(f"Running segmentation for {video_meta}")
    vad = pipeline(audio_path)
    logger.info("Done")

    segments = list(vad.itersegments())
    audio, sample_rate = sf.read(audio_path)
    segment_times = [(segment.start, segment.end) for segment in segments]

    logger.info(f"Extracting audio for {video_meta}")
    result = []
    for i, (start, end) in enumerate(segment_times):
        # Extract the audio
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        audio_clip = audio[start_frame:end_frame]
        wav_path = audio_dir / f"{imdb}_{i}.wav"
        sf.write(wav_path, audio_clip, sample_rate)
        result.append(
            {
                "start": start,
                "end": end,
                "wav_path": str(wav_path),
                "text": None,
            }
        )
    meta_df = pl.DataFrame(result)
    meta_df.write_csv(meta_path, separator="\t")
    logger.info("Done")


def main(args):
    # check if either --video_files or --all is provided
    if not args.video_files and not args.all:
        raise ValueError("Either --video_files or --all must be provided")

    if args.all:
        video_files = list(VideoMetadata.itermovies())
    else:
        video_files = [VideoMetadata.from_video(v) for v in PathList(args.video_files)]

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    model = Model.from_pretrained(MODEL_DIR / "segmentation.bin")
    model.to("cuda")
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate(
        {
            "min_duration_on": 0.5,
            "min_duration_off": 0.01,
        }
    )

    for video_meta in tqdm(video_files):
        extract_audio(video_meta, output_dir, pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_files",
        nargs="+",
        help="The video files to perform speaker segmentation on",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Whether to process all videos in the metadata file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The directory to save the speaker segmentation files to",
        default=DATA_DIR / "asr_extracts",
    )
    args = parser.parse_args()
    main(args)
