"""
Process the spreadsheet data and adds metadata about where the movies are stored
"""

from pathlib import Path

import pandas as pd

from src.utils import DATA_DIR, DEFAULT_META_PATH


def main():
    box_office_df = pd.read_csv(DATA_DIR / "mov_top50.tsv", delimiter="\t")
    box_office_df = box_office_df[box_office_df["in collection?"].eq("Y")]
    awards_df = pd.read_csv(DATA_DIR / "mov_awards.tsv", delimiter="\t").rename(
        columns={"IMDB": "imdb"}
    )
    awards_df = awards_df[awards_df["in collection?"].eq("Y")]
    paths = pd.read_csv(DATA_DIR / "secure_paths.txt", delimiter="\t", names=["path"], header=None)
    paths.loc[:, "ID"] = paths.path.apply(lambda x: Path(x).stem)

    joined = pd.concat([box_office_df, awards_df], axis=0)
    joined = joined.dropna(how="all").reset_index()
    joined = joined[["ID", "imdb", "title", "year"]]
    joined = joined.sort_values(by="year")

    joined = joined.merge(paths, on="ID", how="left")
    joined = joined.drop_duplicates(subset=["ID"], keep="first")
    joined.rename(columns={"ID": "movie_id"}, inplace=True)
    joined.loc[:, "start"] = None
    joined.loc[:, "end"] = None

    joined.to_csv(
        DEFAULT_META_PATH.parent / "secure_metadata.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
