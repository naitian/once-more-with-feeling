from pathlib import Path
from datasets import Dataset, Audio

from src.data.datamodules import create_processor, create_collator


def construct_dataset(
    pathlist: list[Path],
    feature_extractor,
    max_length=128_000,
    sampling_rate=16_000,
):
    """
    Create dataset from the paths
    """
    process = create_processor(feature_extractor)
    dataset = [{"audio": str(path)} for path in pathlist]
    collator = create_collator(feature_extractor, max_length)
    dataset = (
        Dataset.from_list(dataset)
        .cast_column("audio", Audio(sampling_rate))
        .map(process, batched=True, remove_columns=["audio"])
        .select_columns(["input_values"])
    )
    return dataset, collator
