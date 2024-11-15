from contextlib import contextmanager
import csv
import logging
import os
import re
import subprocess
import typing
import unicodedata
from glob import glob
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

import numpy as np
from torchaudio import load


def _setup_dir(env_var, default):
    directory = Path(os.environ.get(env_var, default))
    directory.mkdir(exist_ok=True, parents=True)
    return directory


BASE_DIR = _setup_dir("BASE_DIR", Path(__file__).resolve().parent.parent)
DATA_DIR = _setup_dir("DATA_DIR", BASE_DIR / "data")
LOG_DIR = _setup_dir("LOG_DIR", BASE_DIR / "logs")

# NOTE: not a dir!
DEFAULT_META_PATH = os.environ.get(
    "META_PATH", BASE_DIR / "src" / "video" / "metadata" / "sequoia_metadata.tsv"
)
SHOTS_DIR = Path(os.environ.get("SHOTS_DIR", DATA_DIR / "shots"))
CACHE_DIR = _setup_dir("CACHE_DIR", BASE_DIR / ".cache")
MODEL_DIR = _setup_dir("MODEL_DIR", BASE_DIR / "models")

MOVIE_DIR = Path("/mnt/data1/corpora/movies/")
MELD_DIR = Path("/mnt/data0/corpora/meld/MELD.Raw")
IEMOCAP_DIR = Path("/mnt/data0/corpora/iemocap/IEMOCAP_full_release")


# Set up logger
# TODO: allow setting different log levels through the CLI
def setup_logger():
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(LOG_DIR / "global.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def PathList(path_or_paths, expand_dirs=True):
    """
    `path_or_paths` can be a single path or a list of paths
    Any directories in the paths will be expanded to their immediate children files
    """
    paths = []
    if isinstance(path_or_paths, list):
        paths = path_or_paths
    else:
        paths = [path_or_paths]
    paths = [Path(x) for x in paths]

    # Expand directories
    for i, path in enumerate(paths):
        if expand_dirs and os.path.isdir(path):
            paths[i] = [Path(x) for x in glob(str(path / "*"))]
        else:
            paths[i] = [Path(path)]

    paths = [x for sublist in paths for x in sublist]
    return paths


# TODO: consider moving data utils to a separate file


def load_from_video(video_path):
    """Load audio from a video file into a torch tensor"""
    audio = None
    sample_rate = None
    # first extract audio from the video using ffmpeg into a temp wav file
    with NamedTemporaryFile(suffix=".wav") as f:
        cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 1 -loglevel panic -y {f.name}"
        subprocess.call(cmd, shell=True, stdout=None)
        audio_path = Path(f.name)
        # then load the audio into a torch tensor
        audio, sample_rate = load(str(audio_path))
    return audio, sample_rate


def slugify(text):
    """Convert text to a URL-friendly slug, replacing any punctuation

    From: https://stackoverflow.com/q/5574042
    """
    slug = unicodedata.normalize("NFKD", text)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", slug).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    slug = re.sub(r"^-+|-+$", "", slug)
    return slug


# https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/6
def get_best_cuda():
    result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A5 GPU | grep Free", shell=True
    )
    info = result.decode("utf-8").strip().split("\n")
    available_memory = [int(line.strip().split()[2]) for line in info]
    return np.argmax(available_memory)


# streaming pickle context manager
# usage:
# with spickle.open('file.pkl') as f:
#   for i in iter:
#       f.dump(i)


class PickleStream:
    @staticmethod
    @contextmanager
    def writer(filename: str):
        file = open(filename, "wb")
        try:
            yield pickle.Pickler(file)
        finally:
            file.close()

    @staticmethod
    def read(filename: str):
        file = open(filename, "rb")
        p = pickle.Unpickler(file)
        try:
            while file.peek(1):
                yield p.load()
        finally:
            file.close()


class CsvWriter:
    def __init__(self, file: typing.TextIO, *args, **kwargs):
        self.file = file
        self.fields = None
        self.writer = None
        self.args = args
        self.kwargs = kwargs

    def dump(self, data: dict):
        if self.fields is None:
            self.fields = list(data.keys())
            self.writer = csv.DictWriter(
                self.file, fieldnames=self.fields, *self.args, **self.kwargs
            )
            self.writer.writeheader()
        self.writer.writerow(data)

    @staticmethod
    @contextmanager
    def writer(filename: str, *args, **kwargs):
        file = open(filename, "w", newline="")
        try:
            yield CsvWriter(file, *args, **kwargs)
        finally:
            file.close()

    def close(self):
        self.file.close()
