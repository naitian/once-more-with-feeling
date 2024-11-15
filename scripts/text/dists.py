import argparse
from glob import glob
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import DATA_DIR, CsvWriter


def main(args):
    embedding_files = glob(str(args.embeddings_dir / "*.npy"))
    vec_inds = []
    all_embeddings = []
    for embedding_file in embedding_files:
        path_file = args.embeddings_dir / Path(embedding_file).stem.split("_")[0]
        if not path_file.exists():
            path_file = str(path_file) + ".tsv"
        embeddings = np.load(embedding_file)
        all_embeddings.append(embeddings)

        inds = pd.read_csv(path_file, sep="\t", names=["id", "path"]).id
        vec_inds += list(inds)
    all_embeddings = np.concatenate(all_embeddings)
    n, d = all_embeddings.shape

    print("Creating index with {} elements".format(n))
    nlist = int(np.sqrt(n))

    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    print("Training voronoi")
    sample = all_embeddings[np.random.randint(0, n, 20000)]
    index.train(sample)

    print("Adding embeddings")
    index.add(all_embeddings)
    index.nprobe = 5

    print("Finding neighbors")

    threshold = args.threshold
    radius = 2 - 2 * threshold

    def search_chunk(chunk, chunk_offset):
        lims, dists, inds = index.range_search(chunk, radius)

        # For each utterance, we want to create edges to all other utterances that
        # are within the threshold
        edges = []
        for chunk_ind, (start, end) in enumerate(zip(lims[:-1], lims[1:])):
            i = chunk_ind + chunk_offset
            for offset, j in enumerate(inds[start:end]):
                if j <= i:
                    continue
                cosine_sim = (2 - dists[int(start + offset)]) / 2
                edges.append((vec_inds[i], vec_inds[j], cosine_sim))
        return edges

    # dists = []
    chunk_size = 10_000
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with CsvWriter.writer(args.output_dir / "edges.csv", delimiter="\t") as writer:
        for chunk in tqdm(range(0, n, chunk_size)):
            this_chunk_size = min(chunk_size, n - chunk)
            chunk_dists = search_chunk(
                all_embeddings[chunk : chunk + this_chunk_size], chunk
            )
            for dist in chunk_dists:
                row = {
                    "source": int(dist[0]),
                    "target": int(dist[1]),
                    "distance": dist[2],
                }
                writer.dump(row)
            # dists += chunk_dists

    # write to file
    # out_path = args.output_dir / "edges.csv"
    # out_path.parent.mkdir(exist_ok=True, parents=True)
    # dists = np.array(dists)
    # dists = pl.DataFrame(
    #     {
    #         "source": (dists[:, 0]).astype(int),
    #         "target": (dists[:, 1]).astype(int),
    #         "distance": dists[:, 2],
    #     }
    # )
    # dists.write_csv(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        help="The path to the embedidngs directory",
        default=DATA_DIR / "text_embeddings",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="The threshold for the cosine similarity",
        default=0.8,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The directory to save the distances to",
        default=DATA_DIR / "cluster",
    )
    main(parser.parse_args())
