"""Produces orthographic alignments.

Uses pretrained base wav2vec2 model
"""

import re

import torch
from datasets import Audio, Dataset
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchaudio.functional import forced_align
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils import MODEL_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR / "wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR / "wav2vec2-base-960h").to(device)
model.eval()


def process_audio(row):
    audio = row["audio"]
    processed = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )
    processed["path"] = audio["path"]
    return processed


def preprocess_text(text):
    text = text.upper()
    text = "|".join([word for word in re.split("[ ?,:.]", text) if len(word) > 0])
    return text


def tokenize_chunks(batch):
    """
    This discards the attention masks (since they're not used in w2v2, and adds
    places for off-by-one-errors)
    1. Tokenize each chunk separately.
    2. Concatenate the chunks and record the token-level offsets.
    3. Pad the concatenated tokens.
    4. Return the concatenated tokens and the offsets.
    """
    # import ipdb; ipdb.set_trace()
    sep_token = processor.tokenizer.encode("|")

    def _generator(batch):
        for batch_ind, d in enumerate(batch["text"]):
            for chunk in d:
                yield (batch_ind, preprocess_text(chunk))

    batch_inds, chunks = zip(*_generator(batch))
    tokens = processor.tokenizer(chunks, padding=False)  # {input_ids, attention_mask}
    tokens = tokens["input_ids"]

    def _regroup(tokens, batch_inds):
        curr_batch_ind = 0
        curr_batch = {"input_ids": [], "offsets": []}
        offset = 0
        for batch_ind, input_ids in zip(batch_inds, tokens):
            if batch_ind != curr_batch_ind and curr_batch_ind is not None:
                yield curr_batch
                offset = 0
                curr_batch_ind = batch_ind
                curr_batch = {"input_ids": [], "offsets": []}
            curr_batch["input_ids"] += input_ids + sep_token
            start = offset
            end = offset + len(input_ids)
            offset = end
            curr_batch["offsets"].append((start, end))
        yield curr_batch

    concat = list(_regroup(tokens, batch_inds))
    # FIXME: this is because I converted from batching at collate-time to
    # batching at map-time. This computation should be unnecessary. I should
    # really just change _regroup function such that we don't have to reiterate.
    rotate = {key: [] for key in concat[0]}
    for batch in concat:
        for key in batch:
            rotate[key].append(batch[key])

    return rotate
    # return concat


def align_batch(batch):
    audio = batch["input_values"]
    tokens = batch["input_ids"]
    offsets = batch["offsets"]
    time_offset = (
        model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
    )
    # tokens = batch["tokens"]
    with torch.no_grad():
        output = model(audio.to(device))
        tokens = tokens.to(device)
        # we iterate through each sample in the batch because right now
        # torchaudio forced_align only works with batch_size==1
        log_probs = F.log_softmax(output.logits, dim=-1)
        batch_preds = []
        for toks, probs, offsets in zip(tokens, log_probs, offsets):
            aligned_toks, aligned_probs = forced_align(
                probs.unsqueeze(0),
                toks[toks != 0].unsqueeze(0),
            )
            decoded = processor.tokenizer.decode(
                aligned_toks[0], output_char_offsets=True
            )

            time_offsets = []
            for i, (start, end) in enumerate(offsets):
                start_frame = decoded.char_offsets[start]["start_offset"]
                end_frame = decoded.char_offsets[end - 1]["end_offset"]
                time_offsets.append(
                    (start_frame * time_offset, end_frame * time_offset)
                )
            batch_preds.append(time_offsets)
    return batch_preds


def collate(batch):
    input_values = pad_sequence(
        [d["input_values"].squeeze() for d in batch], batch_first=True
    )
    input_ids = pad_sequence(
        [d["input_ids"].squeeze() for d in batch], batch_first=True
    )
    offsets = [d["offsets"] for d in batch]
    return dict(
        input_values=input_values,
        paths=[d["path"] for d in batch],
        input_ids=input_ids,
        offsets=offsets,
    )


def create_dataset(df, audio_col="audio", text_col="text"):
    """Create compatible dataset from DataFrame

    Each row in the text_col column should be a list of strings. The
    concatenation of these strings forms the entire utterance for the given
    audio; we provide start and end alignments for each string in the list.

    Arguments
    ---------
    df : pd.DataFrame
        DataFrame containing audio and text data
        Should have columns [audio_col, text_col]
    audio_col : str, optional (default="audio")
    text_col : str, optional (default="text")

    Returns
    -------
    Dataset
        Dataset containing audio and text data.
        Has the columns `input_values`, `path`, `input_ids`, and `offsets`
    """
    df = df.dropna(subset=[audio_col, text_col])
    df.loc[:, "audio"] = df[audio_col]
    df.loc[:, "text"] = df[text_col]
    df = df[["audio", "text"]]
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(process_audio)
    dataset = dataset.map(tokenize_chunks, batched=True, batch_size=100)
    dataset = dataset.with_format("torch")
    return dataset.select_columns(["input_values", "path", "input_ids", "offsets"])


def get_alignments(df, audio_col="audio", text_col="text", batch_size=2):
    """Generate alignments for a given dataframe"""
    dataset = create_dataset(df, audio_col, text_col)
    dataloader = DataLoader(dataset, batch_size, collate_fn=collate)
    all_preds = []
    for batch in tqdm(dataloader):
        all_preds += align_batch(batch)
    return all_preds
