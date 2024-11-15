from datetime import datetime
import pandas as pd
from src.utils import DATA_DIR, MELD_DIR, MODEL_DIR

SPLITS = [
    ("train", "train_sent_emo.csv", "train_splits"),
    ("dev", "dev_sent_emo.csv", "dev_splits_complete"),
    ("test", "test_sent_emo.csv", "output_repeated_splits_test"),
]


def coerce_meld_data():
    """
    Make MELD data match the rest of the data

    Have start, end, wav_path, and text columns
    """
    for split, csv_name, audio_dir in SPLITS:
        csv = pd.read_csv(MELD_DIR / csv_name)
        csv = csv.rename(
            columns={
                "Dialogue_ID": "dialogue_id",
                "Utterance_ID": "utterance_id",
                "Utterance": "text",
                "StartTime": "start",
                "EndTime": "end",
            }
        )
        csv.loc[:, "wav_path"] = [
            MELD_DIR
            / audio_dir
            / f"dia{row['dialogue_id']}_utt{row['utterance_id']}.wav"
            for _, row in csv.iterrows()
        ]

        def _parse_time(time):
            parsed = datetime.strptime(time, "%H:%M:%S,%f")
            ref = datetime.strptime("00:00:00,000", "%H:%M:%S,%f")
            return (parsed - ref).total_seconds()

        csv.loc[:, "start"] = (
            csv["start"].apply(_parse_time) + csv["Episode"] * 60 * 60 * 24
        )
        csv.loc[:, "end"] = (
            csv["end"].apply(_parse_time) + csv["Episode"] * 60 * 60 * 24
        )
        csv.to_csv(
            DATA_DIR / "datasets" / "meld" / split / f"{split}.tsv",
            index=False,
            sep="\t",
        )


def group_convos():
    from src.data.group_convos import group_conversations

    for split, csv_name, _ in SPLITS:
        csv_path = DATA_DIR / "datasets" / "meld" / split / f"{split}.tsv"
        grouped = group_conversations(csv_path, min_gap=3)  # 3 seconds minimum gap
        grouped.to_csv(csv_path, sep="\t", index=False)


def make_embeddings():
    from src.audio.embeddings import main as create_embeddings

    for split, csv_name, audio_dir in SPLITS:
        dataset_path = DATA_DIR / "datasets" / "meld" / split
        dataset_path.mkdir(exist_ok=True, parents=True)
        csv = pd.read_csv(MELD_DIR / csv_name)
        pathlist = [
            MELD_DIR
            / audio_dir
            / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
            for _, row in csv.iterrows()
        ]
        pathlist = [p for p in pathlist if p.exists()]

        create_embeddings(
            audio_path=pathlist,
            model_path=MODEL_DIR / "wav2vec2-large-lv60",
            output_path=dataset_path / f"{split}.pkl",
        )


if __name__ == "__main__":
    coerce_meld_data()
    group_convos()
    make_embeddings()
