"""
(Hopefully performant) I/O utilities for video files
"""

import subprocess
from pathlib import Path

from src.utils import logger, CACHE_DIR



def audio_file(video_path: Path, sampling_rate=16000):
    """
    Extract the audio file to a cache file and return the path to the cache file
    if the file does not exist, otherwise return the path to the cache file.
    """
    outfile = CACHE_DIR / f"{video_path.stem}.wav"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    if not outfile.exists():
        logger.info(f"Extracting audio from {video_path} to {outfile}")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sampling_rate),
                "-ac",
                "1",
                "-f",
                "wav",
                "-y",
                str(outfile.resolve()),
            ],
            check=True,
        )
        logger.info("Done")
    return outfile
