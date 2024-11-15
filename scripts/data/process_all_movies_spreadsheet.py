"""
Process the spreadsheet data and adds metadata about where the movies are stored

Merges screen_capture.tsv and movies.tsv
"""

import polars as pl

from src.utils import DATA_DIR, DEFAULT_META_PATH


def main():
    movie_df = pl.read_csv(DATA_DIR / "movies.tsv", has_header=True, separator="\t")
    movie_df = movie_df.with_columns(
        start=pl.lit(None), end=pl.lit(None), path=pl.lit("/fake/path/for/movies.mp4")
    )
    joined = movie_df.select(
        pl.col("ID", "imdb", "title", "year", "path", "start", "end")
    ).rename({"ID": "movie_id"})

    # Dedupe
    joined = joined.unique("imdb")
    joined.write_csv(
        DEFAULT_META_PATH.parent / "all_movies_metadata.tsv", separator="\t"
    )


if __name__ == "__main__":
    main()
