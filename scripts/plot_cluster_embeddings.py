#!/usr/bin/env python3
"""
plot_cluster_embeddings.py

Visualise clustered document embeddings while
  • reading the mapping from code.json          (--json)
  • taking legend order from cluster_analysis.json (--order-json)
  • hiding clusters with <10 docs (after intersection with embeddings)
  • skipping UNCLASSIFIED_NOISE and MISSING_ID_METADATA
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --------------------------------------------------------------------------- #
MIN_CLUSTER_SIZE = 10            # hard-coded threshold
IGNORED = {"UNCLASSIFIED_NOISE", "MISSING_ID_METADATA"}
# --------------------------------------------------------------------------- #


def load_embeddings(parq: Path):
    df = pd.read_parquet(parq)
    if "vector" in df.columns:                     # common pattern: a list column
        X = np.vstack(df["vector"].to_list())
    else:                                         # else assume all non-id cols are dims
        dims = [c for c in df.columns if c != "doc_id"]
        X = df[dims].values
    ids = df["doc_id"].astype(str).tolist()
    logging.info("Embeddings: %d docs, %d dims", len(ids), X.shape[1])
    return ids, X


def load_mapping(mapping_json: Path) -> dict[str, str]:
    """
    Build dict {doc_id -> cluster_id} from the code.json structure you provided.
    """
    data = json.loads(mapping_json.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}

    def ingest(section: dict[str, dict]):
        for cid, info in section.items():
            if cid in IGNORED:
                continue                                       # skip noise/missing
            for doc_id in info.get("document_ids", []):
                mapping[str(doc_id)] = cid

    ingest(data.get("final_categories", {}))
    ingest(data.get("provisional_candidate_groups", {}))
    logging.info("Mapping loaded: %d docs across %d clusters",
                 len(mapping), len(set(mapping.values())))
    return mapping


def reduce_to_2d(X: np.ndarray, method: str = "pca"):
    if method == "pca":
        return PCA(n_components=2).fit_transform(X)
    if method == "tsne":
        return TSNE(n_components=2, init="random", random_state=42).fit_transform(X)
    raise ValueError("--method must be 'pca' or 'tsne'")


def plot(coords: np.ndarray, labels: list[str],
         legend_order: list[str], outfile: Path | None):
    n = len(legend_order)                                   # ← NEW
    cmap_disc = cm.get_cmap('Set1', n)                      # ← NEW

    colour_map = {cid: i for i, cid in enumerate(legend_order)}
    colours = [colour_map[c] for c in labels]

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1],
                c=colours,
                cmap=cmap_disc,      # ← use the discrete cmap
                vmin=0, vmax=n-1,    # ← keep indices discrete
                s=18)

    handles = [
        Line2D([0], [0], marker='o',
               color=cmap_disc(i),   # ← pull exact colour i from the cmap
               linestyle='', markersize=6, label=cid)
        for i, cid in enumerate(legend_order)
    ]
    plt.legend(handles=handles, title="Cluster",
               bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    plt.title("Document Embedding Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=300)
        logging.info("▶ saved → %s", outfile)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, type=Path,
                    help="doc_vectors.parquet")
    ap.add_argument("--json", required=True, type=Path,
                    help="code.json (doc→cluster mapping)")
    ap.add_argument("--order-json", required=True, type=Path,
                    help="cluster_analysis.json (desired legend order)")
    ap.add_argument("--method", choices=("pca", "tsne"), default="pca")
    ap.add_argument("--output", type=Path,
                    help="PNG to write (omit to show interactively)")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.INFO)

    # 1. Load data
    doc_ids, X = load_embeddings(args.emb)
    mapping = load_mapping(args.json)

    # 2. Order list for legend
    order_list = json.loads(args.order_json.read_text(encoding="utf-8"))
    legend_order = [
        e["cluster_id"] for e in order_list
        if e["cluster_id"] not in IGNORED
    ][:7]   # ← keep only the first 7 clusters in the specified order

    # 3. Intersect embeddings ↔ mapping
    pairs = [(vec, mapping[d]) for d, vec in zip(doc_ids, X) if d in mapping]
    if not pairs:
        logging.error("No doc IDs overlap between embeddings and mapping.")
        return

    vecs, labs = zip(*pairs)
    present_counts = Counter(labs)
    logging.info("Docs per cluster (present): %s", dict(present_counts))

    # 4. Keep clusters with ≥ MIN_CLUSTER_SIZE
    valid_cids = [cid for cid in legend_order if present_counts.get(cid, 0) >= MIN_CLUSTER_SIZE]
    if not valid_cids:
        logging.error("No cluster has ≥ %d docs present. Adjust threshold?",
                      MIN_CLUSTER_SIZE)
        return

    X_f, y_f = zip(*[(v, l) for v, l in zip(vecs, labs) if l in valid_cids])
    X_f = np.vstack(X_f)
    logging.info("Plotting %d clusters (%d points total)",
                 len(valid_cids), len(y_f))

    # 5. Dim reduction & plot
    coords = reduce_to_2d(X_f, args.method)
    plot(coords, y_f, valid_cids, args.output)


if __name__ == "__main__":
    main()
