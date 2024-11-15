from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import DATA_DIR

dataset = pd.read_json(DATA_DIR / "annotation_data/data.ndjson", lines=True)
outdir = DATA_DIR / "annotation_data/audio_clips"
outdir.mkdir(exist_ok=True, parents=True)


def get_audios_for_clip(convo):
    df = pd.read_csv(
        DATA_DIR / "asr_extracts" / "split_data" / f"{convo.imdb}.tsv", sep="\t"
    )
    start_clip = df[df["start"] == convo.start]
    start_ind = start_clip.index[0]
    num_utterances = len(convo.utterances)
    clips = df.iloc[start_ind : start_ind + num_utterances]

    # symlink clip path to output dir
    for _, clip in clips.iterrows():
        clip_path = clip.wav_path
        out_path = outdir / f"{Path(clip.wav_path).name}"
        try:
            out_path.symlink_to(clip_path)
        except FileExistsError:
            pass

    return clips.wav_path.to_list()


dataset["audio_clips"] = None
dataset["audio_clips"] = dataset["audio_clips"].astype("object")
for i, clip in tqdm(dataset.iterrows(), total=len(dataset)):
    dataset.at[i, "audio_clips"] = get_audios_for_clip(clip)

dataset.to_json(
    DATA_DIR / "annotation_data/data_audio.ndjson", lines=True, orient="records"
)
