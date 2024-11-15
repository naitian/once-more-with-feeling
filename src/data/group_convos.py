"""
Group contiguous chunks of utterances. This is used both for the annotation data
as well as for the contextual emotion model.
"""

import argparse
from pathlib import Path

import pandas as pd


def group_conversations(data: Path, min_gap=5):
    transcript = pd.read_csv(data, delimiter="\t")
    transcript.loc[:, "group"] = (
        (transcript.start.shift(-1) - transcript.end)
        .gt(min_gap)
        .cumsum()
        .shift()
        .fillna(0)
        .astype(int)
    )
    return transcript


def main(data: Path, min_gap=5):
    transcript = group_conversations(data, min_gap)
    transcript.to_csv(
        data.parent / f"{data.stem()}_grouped.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        type=Path,
        help="The path to the data file containing the utterances.",
    )
    parser.add_argument(
        "--min_gap",
        type=int,
        help="The minimum gap between clusters in seconds.",
        default=3,
    )
    args = parser.parse_args()
    main(args.data, args.min_gap)
