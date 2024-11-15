"""
Generate a file with the paths for a given cluster
"""

import argparse
from pathlib import Path
import polars as pl

from tqdm import tqdm

from src.utils import slugify


# TODO: import this from the appropriate place
def load_cluster(cluster_path):
    clusters = {}
    for line in tqdm(open(cluster_path, "r")):
        if line.startswith("node"):
            continue
        node, cluster = line.strip().split(",")
        clusters[int(node)] = [int(x) for x in cluster.split()]
    return clusters


def main(args):
    clusters = load_cluster(args.cluster_dir / "cluster.csv")
    srt_texts = pl.read_csv(args.cluster_dir / "texts.csv", has_header=True)

    ind_to_cluster = {
        i: cluster for cluster, nodes in clusters.items() for i in nodes
    }

    if args.all:
        paths = srt_texts.with_row_index().with_columns(
            pl.col("index").replace(ind_to_cluster, default=-1).alias("cluster")
        )
        output_path = args.cluster_dir / "paths" / "all.csv"

    else:
        if args.cluster is None:
            raise ValueError("You must specify a cluster if --all is not set")
        if args.cluster not in clusters:
            raise ValueError(f"Cluster {args.cluster} not found in {args.cluster_dir}")


        cluster = clusters[args.cluster]
        paths = srt_texts[cluster]
        example = slugify(srt_texts["text"][cluster[0]])[:10]
        output_path = args.cluster_dir / "paths" / f"{args.cluster}-{example}.csv"

    output_path.parent.mkdir(exist_ok=True, parents=True)
    if args.path_only:
        paths = paths.select(pl.col("wav_path"))
    paths.write_csv(output_path, separator="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_dir",
        type=Path,
        help="The path to the directory containing the cluster file",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        help="The cluster to generate the paths for",
    )
    parser.add_argument(
        "--path_only",
        action="store_true",
        help="If set, only the path to the audio file is written to the output file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If set, generate paths for all clusters in the directory",
    )
    args = parser.parse_args()
    main(args)
