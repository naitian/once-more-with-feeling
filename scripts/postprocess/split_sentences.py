"""
Split multi-sentence audios into individual sentences.
"""

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm
from syntok import segmenter

from src.utils import logger
from src.video.meta import VideoMetadata
from src.audio.alignment import get_alignments


def segment_sentences(doc):
    for paragraph in segmenter.process(doc):
        for sent in paragraph:
            start, end = (sent[0].offset, sent[-1].offset + len(sent[-1].value))
            yield doc[start:end]


def main(args):
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.all is None and args.imdb_ids is None:
        raise ValueError("Please provide either --all or --imdb_ids")
    if args.all:
        imdb_ids = [x.imdb for x in VideoMetadata.itermovies()]
    else:
        imdb_ids = args.imdb_ids

    for imdb_id in imdb_ids:
        logger.info("Processing %s", imdb_id)
        text_data = pd.read_csv(
            Path("data/asr_extracts/data") / f"{imdb_id}.tsv", sep="\t"
        )
        text_data = text_data.dropna(subset=["wav_path", "text"])
        # filter to audios that have multiple sentences
        text_data.loc[:, "sentences"] = text_data.text.apply(segment_sentences).apply(
            list
        )
        text_data.loc[:, "num_sentences"] = text_data.sentences.apply(len)
        subset = text_data[text_data.num_sentences > 1].copy()
        logger.info(
            f"There are {len(text_data)} rows in the dataset of which {len(subset)} have multiple sentences. Aligning..."
        )
        if len(subset) == 0:
            logger.info("No sentences to split. Moving on...")
            continue
        subset.loc[:, "alignments"] = get_alignments(
            subset, audio_col="wav_path", text_col="sentences", batch_size=2
        )
        logger.info("Alignments complete. Splitting sentences...")

        # create new audio files for the sentences
        split_subset = []
        for i, row in tqdm(subset.iterrows(), total=len(subset)):
            wav_path = Path(row.wav_path)
            audio, sr = sf.read(row.wav_path)
            for j, sentence in enumerate(row.sentences):
                start, end = row.alignments[j]
                slice = audio[int(start * sr) : int(end * sr) + 1]
                output_path = f"{wav_path.parent / wav_path.stem}_{j}.wav"
                sf.write(output_path, slice, sr)
                split_subset.append(
                    {
                        "wav_path": output_path,
                        "text": sentence,
                        "start": row.start + start,
                        "end": row.start + end,
                    }
                )

        logger.info(
            f"Writing new dataset. There were {len(subset)} utterances with multiple sentences. This has been split into {len(split_subset)} utterances."
        )
        split_subset = pd.DataFrame(split_subset)
        text_data = text_data.drop(subset.index)
        text_data = pd.concat([text_data, split_subset], ignore_index=True)
        text_data = text_data.drop(columns=["sentences", "num_sentences"])
        text_data = text_data.sort_values(by="start")
        text_data.to_csv(
            args.output_dir / f"{imdb_id}.tsv", sep="\t", index=False
        )
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imdb_ids", nargs="+", help="The IMDb IDs of the movies to align"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Whether to process all movies in the metadata file",
    )
    parser.add_argument(
        "--extracts_dir",
        type=Path,
        help="The directory containing the audio and text extracts",
        default=Path("data/asr_extracts"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The directory to save the new data files.",
        default=Path("data/asr_extracts/split_data")
    )
    args = parser.parse_args()
    main(args)
