"""
Generate BERT embeddings for the text utterances.
"""

import argparse
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from src.utils import MODEL_DIR, get_best_cuda


def main(args):
    device = f"cuda:{get_best_cuda()}"
    model_name = MODEL_DIR / "all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # tokenize step -- for now, we replace any null text with an empty string.
    # In the future, for the sake of efficiency, we should remove the null text
    # rows (as well as any empty string rows)
    texts = pd.read_csv(args.text_path, sep="\t", names=["id", "text"])
    tokens = None
    outputs = []
    for batch in tqdm(DataLoader(texts["text"].to_list(), batch_size=1000)):
        output = tokenizer(batch, max_length=128, truncation=True)
        outputs.append(output)
    tokens = {
        key: list(chain.from_iterable([output[key] for output in outputs]))
        for key in outputs[0]
    }

    dataset = [
        {key: tokens[key][i] for key in tokens} for i in range(len(tokens["input_ids"]))
    ]

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    if args.output_dir is None:
        args.output_dir = args.text_path.parent
    args.output_dir.mkdir(exist_ok=True, parents=True)
    all_embeddings = np.zeros((len(texts), 768))
    for i, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            output = model(**batch.to(device))
            this_batch_size = len(batch["input_ids"])
            embeddings = output.pooler_output.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
            all_embeddings[i * batch_size : i * batch_size + this_batch_size] = (
                embeddings
            )
    np.save(args.output_dir / f"{args.text_path.stem}_embeddings.npy", all_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text_path",
        type=Path,
        help="Path to the file containing text utterance (output or split output of ingest_text.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the directory to save the embeddings",
        default=None,
    )
    args = parser.parse_args()
    main(args)
