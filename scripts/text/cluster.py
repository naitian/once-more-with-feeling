"""
Run leiden clustering
"""

import argparse
from pathlib import Path

import igraph as ig
import leidenalg as la
from tqdm import tqdm

from src.utils import DATA_DIR, logger


def read_graph_from_edges(edges_path):
    def read_edges(edges_path):
        for line in open(edges_path, "r"):
            if line.startswith("s"):
                continue
            source, target, weight = line.strip().split("\t")
            yield int(source), int(target), float(weight)

    packed = (
        ((source, target), weight) for source, target, weight in read_edges(edges_path)
    )
    edges, weights = zip(*packed)
    return edges, weights


def load_cluster(cluster_path):
    clusters = {}
    for line in tqdm(open(cluster_path, "r")):
        if line.startswith("node"):
            continue
        node, cluster = line.strip().split(",")
        clusters[node] = [int(x) for x in cluster.split()]
    return clusters


def save_cluster(clusters, cluster_path):
    with open(cluster_path, "w") as f:
        f.write("node,cluster\n")
        for node, cluster in tqdm(clusters.items()):
            f.write(f"{node},{' '.join(str(x) for x in cluster)}\n")


def main(args):
    logger.info("Reading edges graph")
    edges, weights = read_graph_from_edges(args.edge_path)
    logger.info("Constructing graph")
    graph = ig.Graph(edges=edges, directed=False, edge_attrs={"weight": weights})

    logger.info("Finding partitions")
    partition = la.find_partition(graph, la.ModularityVertexPartition)

    outfile = args.output_dir / "cluster.csv"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    clusters = {node: cluster for node, cluster in enumerate(partition)}
    save_cluster(clusters, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_path",
        type=Path,
        help="The path to the edges file",
        default=DATA_DIR / "embeddings/edges.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The directory to save the clusters to",
        default=DATA_DIR / "embeddings",
    )
    main(parser.parse_args())
