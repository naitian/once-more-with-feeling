"""
Data utilities for utterances extracted from the SRTs

TODO: rename this because we're reading the extracted TSVs, not the SRTs themselves.
"""

import polars as pl

from src.utils import DATA_DIR

TEXT_DIR = DATA_DIR / "audio_extracts" / "data"


def load_srt_text(pathlist) -> str:
    """
    Load the text from the SRT file for the given IMDB ID.

    Args:
        imdb (str): IMDB ID of the movie

    Returns:
        str: Text from the SRT file
    """
    csvs = []
    for path in pathlist:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        try:
            df = pl.read_csv(path, has_header=True, separator="\t")
        except pl.exceptions.NoDataError:
            continue
        df = df.with_columns(imdb_id=pl.lit(path.stem))
        csvs.append(df)
    return pl.concat(csvs)
