"""
Video metadata utils for video data
"""

from pathlib import Path

import polars as pl

from src.utils import DEFAULT_META_PATH, SHOTS_DIR


class VideoMetadata:
    meta_df = pl.read_csv(DEFAULT_META_PATH, has_header=True, separator="\t")

    def __init__(self, movie_id, imdb, title, year, path, start, end):
        self.movie_id = movie_id
        self.imdb = imdb
        self.title = title
        self.year = year
        self.path = Path(path)
        self.start = start
        self.end = end

        self._shots = None

    def __repr__(self):
        return f"VideoMetadata({self.movie_id} - {self.imdb})"

    @property
    def shots(self):
        if self._shots is None:
            shot_file = SHOTS_DIR / f"{self.path.name}.scenes.txt"
            if not shot_file.exists():
                raise ValueError(f"No shot file found for {self.path}")
            with open(shot_file) as f:
                self._shots = [tuple(map(int, line.strip().split(" "))) for line in f]
        return self._shots

    @classmethod
    def _find_row(cls, exp):
        results = cls.meta_df.filter(exp)
        if len(results) == 0:
            return None
        if len(results) > 1:
            raise ValueError(f"Multiple rows found for {exp}")
        return results.to_dicts()[0]

    @classmethod
    def from_video(cls, video_path: Path):
        data = cls._find_row(pl.col("path") == str(video_path))
        if data is None:
            raise ValueError(f"No metadata found for {video_path}")
        return cls(**data)

    @classmethod
    def from_imdb(cls, imdb_id: str):
        data = cls._find_row(pl.col("imdb") == imdb_id)
        if data is None:
            raise ValueError(f"No metadata found for {imdb_id}")
        return cls(**data)

    @classmethod
    def itermovies(cls):
        for row in cls.meta_df.iter_rows(named=True):
            yield cls(**row)
