# Data processing scripts

## Metadata files

`screen_capture.tsv` is the list of movies we have screencaps of on sequoia; `movies.tsv` is the master document with all of the movies in our collection

## Preprocessing metadata

These scripts create standardized files with the following fields: `movie_id` (a unique movie identifier from the spreadsheet), `imdb` (the IMDB ID), `title`, `year`, `path` (local path to the movie), `start`, `end` (where the start and end are optional fields for start and end timestamps if there is extra time before or after the movie)

- `process_spreadsheet.py` prepares the `screen_capture.tsv` file, outputs `sequoia_metadata.tsv`
- `process_all_movies_spreadsheet.py` does the same but for `movies.tsv`, outputs `all_movies_metadata.tsv`


## SRT alignment and preprocessing

One way to get text / audio alignments is to use the subtitle files from OpenSubtitles and align the timestamps. To do this, run:

1. `get_srts.py`: this file fetches the relevant SRT files from our subtitle corpus and applies the relevant timestamp adjustments
2. `run_ffsubsync.py`: runs `ffsubsync` to get audio-aligned subtitles
3. `clean_srt.py`: cleans up SRT files (removes music, italics, etc)
4. `extract_audio.py`: uses the cleaned, aligned SRT files to extract audio snippets that are aligned to the text utterances

Note: `extract_dummy.py` does something similar to `extract_audio.py`, except it only touches the SRT files and does nothing with the audio. This is useful for extracting the utterances for analysis (e.g. clustering) but you don't have access to the actual movie files.

## ASR alignment

1. `segment.py`: retrieves speech segments from the video
2. `transcribe.py`: transcribes each individual speech segment

The output has the same structure as `extract_audio.py` from using SRT alignment.

## Text and audio post-processing

1. `split_sentences.py`: uses wav2vec2 ASR model to split any utterances with multiple sentences (NOTE: this script modifies the data in-place!)

## Emotion recognition

We process in batches of audio files
```sh
find data/asr_extracts/ -name "*.wav" > data/all_audio_paths.txt
split -l 2000 data/all_audio_paths.txt data/audio_path_txt/
for i in $(ls data/audio_path_txt/); do  python src/run_inference.py speech-emotion/7g1ygbvx/checkpoints/epoch\=1-step\=626.ckpt  data/audio_path_txt/$i --output_dir=data/inference_output/all/$i; done
```

We can precompute word2vec2 embeddings of utterances using `src/audio/embeddings.py`

## Text embedding

We embed and cluster the extracted texts.

1. `embeddings.py`: gets the sentence embeddings
2. `dists.py`: computes the distances between texts for the clustering step
3. `cluster.py`: clusters the embeddings
