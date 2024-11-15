"""
Process the spreadsheet data and adds metadata about where the movies are stored

Merges screen_capture.tsv and movies.tsv
"""

from pathlib import Path

import pandas as pd

from src.utils import DATA_DIR, DEFAULT_META_PATH

TEST_MOVIE_DIR = Path("/global/home/groups/isch-aux01-access/movies/")


imdb_ids = [
    "tt0407887",
    "tt0091042",
    "tt2278388",
    "tt0088847",
    "tt0362270",
    "tt1748122",
    "tt0093748",
    "tt0265666",
    "tt0128445",
    "tt0088128",
    "tt0090305"
]


def main():
    screencap_df = pd.read_csv(DATA_DIR / "screen_capture.tsv", delimiter="\t")

    subset = screencap_df[screencap_df.IMDB.isin(imdb_ids)]
    subset["path"] = subset.title.apply(lambda x: str(TEST_MOVIE_DIR / f"{x}.mov"))
    # First, fill in any missing paths for screencap_df
    subset.loc[:, "movie_id"] = subset.title
    subset.loc[:, "year"] = 1900  # set fake year
    subset = subset.rename(columns={"IMDB": "imdb"})

    subset.to_csv(DEFAULT_META_PATH.parent / "secure_test_metadata.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()