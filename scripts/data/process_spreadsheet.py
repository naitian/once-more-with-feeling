"""
Process the spreadsheet data and adds metadata about where the movies are stored

Merges screen_capture.tsv and movies.tsv
"""

from pathlib import Path

import polars as pl

from src.utils import DATA_DIR, DEFAULT_META_PATH

SEQUIOA_MOVIE_DIR = Path("../movie-representations/movie_reps/data/movies/")


def main():
    movie_df = pl.read_csv(DATA_DIR / "movies.tsv", has_header=True, separator='\t')
    screencap_df = pl.read_csv(DATA_DIR / "screen_capture.tsv", has_header=True, separator='\t')

    # First, fill in any missing paths for screencap_df
    screencap_df = screencap_df.with_columns(pl.col("path").fill_null(pl.col("title").map_elements(lambda x: str(SEQUIOA_MOVIE_DIR / x))))
    screencap_df = screencap_df.filter(pl.col("path").map_elements(lambda x: Path(x).exists()))

    # Merge the two dataframes on the imdb id. This is `IMDB` in screencap_df and `imdb` in movie_df
    # There are some screen capture movies that are not in the movie_df. This is fine, we can just ignore them
    joined = movie_df.join(screencap_df, left_on="imdb", right_on="IMDB", how="inner")
    joined = joined.select(pl.col("ID", "imdb", "title", "year", "path", "start", "end")).rename({"ID": "movie_id"})

    # Dedupe
    joined = joined.unique("imdb")
    joined.write_csv(DEFAULT_META_PATH.parent / "sequoia_metadata.tsv", separator="\t")


if __name__ == "__main__":
    main()