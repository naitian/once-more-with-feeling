"""
Create huggingface dataset for MELD dataset

NOTE: dia125_utt3 is corrupted; we will skip it
"""

from datasets import Dataset, Audio, DatasetDict, ClassLabel

from src.utils import DATA_DIR, MELD_DIR


SPLITS = [
    ("train", "train_sent_emo.csv", "train_splits"),
    ("dev", "dev_sent_emo.csv", "dev_splits_complete"),
    ("test", "test_sent_emo.csv", "output_repeated_splits_test"),
]


def add_audio(example, audio_dir):
    audio_path = (
        MELD_DIR
        / audio_dir
        / f"dia{example['Dialogue_ID']}_utt{example['Utterance_ID']}.wav"
    )
    if not audio_path.exists():
        print(f"Skipping {audio_path} because it doesn't exist")
        example["audio"] = None
        return example
    example["audio"] = str(audio_path)
    return example


def main():
    dataset = DatasetDict()
    for split in SPLITS:
        split_name, csv_name, audio_dir = split
        dataset[split_name] = Dataset.from_csv(str(MELD_DIR / csv_name))
        dataset[split_name] = dataset[split_name].map(
            add_audio, fn_kwargs=dict(audio_dir=MELD_DIR / audio_dir)
        ).filter(lambda x: x["audio"] is not None)
        dataset[split_name] = dataset[split_name].cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
        dataset[split_name] = dataset[split_name].cast_column(
            "Emotion",
            ClassLabel(
                names=[
                    "neutral",
                    "joy",
                    "sadness",
                    "anger",
                    "fear",
                    "disgust",
                    "surprise",
                ]
            ),
        )

    dataset.save_to_disk(DATA_DIR / "meld")


if __name__ == "__main__":
    main()
