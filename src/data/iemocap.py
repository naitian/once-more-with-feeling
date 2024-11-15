"""
Create huggingface dataset with IEMOCAP dataset
"""

import argparse

import polars as pl
from datasets import Audio, ClassLabel, Dataset, DatasetDict

from src.utils import DATA_DIR, IEMOCAP_DIR


def build_iemocap(train_sessions, val_sessions, squash_emotions=False):
    """
    Create huggingface dataset for IEMOCAP dataset.

    The dataset will be split into train and val splits according to the sessions.

    If `squash_emotions` is True, the emotions will be squashed into 4 classes,
    where happy and excited are combined into one class.
    """
    dataset = DatasetDict()
    df = pl.read_csv(DATA_DIR / "iemocap.csv")
    df = df.with_columns(
        pl.col("FileName").str.extract(r"Ses(\d+)*").str.to_integer().alias("Session")
    )
    # we do a dumb reverse, slice, reverse bc polars doesn't currently support
    # negative end indices
    # see here: https://github.com/pola-rs/polars/issues/7127
    # functionally what we're getting is pl.col(...).str.split("_").list.slice(-1)
    df = df.with_columns(
        pl.col("FileName")
        .str.split("_")
        .list.reverse()
        .list.slice(1)
        .list.reverse()
        .list.join("_")
        .alias("Folder")
    )

    class_names = ["neu", "ang", "hap", "exc", "sad"]
    

    if squash_emotions:
        df = df.with_columns(
            pl.when(pl.col("Label").is_in(["hap", "exc"]))
            .then(pl.lit("hap"))
            .otherwise(pl.col("Label"))
            .alias("Label")
        )
        class_names = ["neu", "ang", "hap", "sad"]

    def get_items_for_split(split):
        sessions = train_sessions if split == "train" else val_sessions

        examples = df.filter(pl.col("Session").is_in(sessions))
        for row in examples.iter_rows(named=True):
            example = {
                "audio": str(
                    IEMOCAP_DIR
                    / f"Session{row['Session']}"
                    / "sentences"
                    / "wav"
                    / row["Folder"]
                    / f"{row['FileName']}.wav"
                ),
                "label": row["Label"],
                "text": row["Sentences"],
            }
            yield example

    for split in ["train", "dev"]:
        dataset[split] = (
            Dataset.from_generator(get_items_for_split, gen_kwargs={"split": split})
            .cast_column("audio", Audio(sampling_rate=16_000))
            .cast_column(
                "label",
                ClassLabel(names=class_names),
            )
        )
    return dataset


def main(args):
    """Build and save the IEMOCAP dataset"""
    dataset = build_iemocap(args.train_sessions, args.val_sessions, args.squash)
    dataset_name = [
        "iemocap",
        "".join(map(str, args.train_sessions)),
        "".join(map(str, args.val_sessions)),
    ]
    if args.squash:
        dataset_name.append("squashed")
    dataset_name = "-".join(dataset_name)
    dataset.save_to_disk(DATA_DIR / dataset_name)


if __name__ == "__main__":
    # Just for testing; you should import build_iemocap in another script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-sessions",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="The sessions to use for training",
    )
    parser.add_argument(
        "--val-sessions",
        type=int,
        nargs="+",
        default=[5],
        help="The sessions to use for validation",
    )
    parser.add_argument(
        "--squash",
        action="store_true",
        help="Squash emotions into 4 classes",
    )
    args = parser.parse_args()
    main(args)
