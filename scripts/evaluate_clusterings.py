# evaluate_clusterings.py  
# ───────────────────────────────────────────────────────────────
"""
Compare multiple document‑level clusterings and write a single Excel report.

What it measures for **each** clustering:
  • Silhouette score (cosine)                    ↑ better
  • Davies–Bouldin Index (Euclidean)             ↓ better
  • #clusters (incl. noise if any)
  • #tiny clusters   (≤ --tiny_thresh docs)      ↓ better
  • #huge clusters   (≥ --huge_frac ⋅ N docs)    ↓ better
  • Full cluster‑size histogram (written to its own sheet)

Inputs
------
  --emb  doc_vectors_*.parquet    (pandas‑Arrow table, cols: doc_id, vector)
  --clusters  one or more pairs  <csv_path>:<label_col>
      Ex:   --clusters  meta_aleph.csv:meta_cluster  meta_e5.csv:meta_cluster  \
                         fuse_eac_hdb.csv:cluster  fuse_km.csv:cluster
  All CSV files must contain at least the given label column and `doc_id`.

Output
------
  An Excel workbook (default: clustering_eval.xlsx) with:
    • Sheet "summary"  – one row per clustering with the metrics above
    • One sheet per clustering (named after the file stem) listing cluster sizes
      (columns: cluster, size).

Example
-------
python evaluate_clusterings.py \  
       --emb  doc_vectors_e5.parquet \
       --clusters \
           doc_meta_clusters_aleph.csv:meta_cluster \
           doc_meta_clusters_e5.csv:meta_cluster \
           fuse_eac_final_with_dist.csv:cluster \
           fuse_eac_km.csv:cluster \
           fused_hier_avg_k15.csv:cluster \
           fused_hier_avg_k19.csv:cluster \
       --tiny_thresh 3 --huge_frac 0.20 \
       --out clustering_eval.xlsx
"""

import argparse, re, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", type=Path, required=True,
                   help="Parquet with doc_id, vector (doc‑level embeddings)")
    p.add_argument("--clusters", nargs="+", required=True,
                   help="CSV:col pairs, e.g. clustering.csv:cluster_col")
    p.add_argument("--tiny_thresh", type=int, default=3,
                   help="Docs ≤ this count are considered 'tiny' clusters")
    p.add_argument("--huge_frac", type=float, default=0.20,
                   help="Fraction of corpus that marks a cluster as 'huge'")
    p.add_argument("--out", type=Path, default=Path("clustering_eval.xlsx"))
    return p.parse_args()

# ── helpers ─────────────────────────────────────────────────────

def load_embeddings(parquet_path: Path):
    tbl   = pq.read_table(parquet_path)
    vecs  = np.stack(tbl["vector"].to_numpy())
    docid = tbl["doc_id"].to_numpy().astype(str)
    return docid, vecs


def load_labels(csv_path: Path, col: str, doc_ids):
    df = pd.read_csv(csv_path, dtype={"doc_id": str})
    if col not in df.columns:
        raise KeyError(f"{csv_path} lacks column '{col}'")
    series = df.set_index("doc_id")[col]
    # align to embedding order
    return series.reindex(doc_ids).to_numpy()

# ── main evaluation ─────────────────────────────────────────────

def evaluate(label_arr, X, tiny_thr, huge_frac):
    # internal metrics
    sil = silhouette_score(X, label_arr, metric="cosine")
    dbi = davies_bouldin_score(X, label_arr)  # Euclidean inside

    # cluster size stats
    counts = Counter(label_arr)
    tiny = sum(1 for c in counts.values() if c <= tiny_thr)
    huge = sum(1 for c in counts.values() if c >= huge_frac * len(label_arr))
    return sil, dbi, counts, tiny, huge

# ── entry ───────────────────────────────────────────────────────

def main():
    args = parse_args()

    doc_ids, X = load_embeddings(args.emb)
    print(f"Loaded {len(doc_ids):,} doc embeddings from {args.emb}")

    summary_rows = []  # for the summary sheet
    size_sheets = {}   # sheet_name -> DataFrame for cluster sizes

    for pair in args.clusters:
        if ":" not in pair:
            print(f"[warn] Skipping '{pair}' – needs format CSV:col", file=sys.stderr)
            continue
        csv_path, col = pair.split(":", 1)
        csv_path = Path(csv_path)
        name = csv_path.stem  # sheet name / summary name

        labels = load_labels(csv_path, col, doc_ids)
        print(f"→ {name}: loaded labels, n_clusters≈{len(set(labels))}")

        sil, dbi, counts, tiny, huge = evaluate(labels, X,
                                                args.tiny_thresh, args.huge_frac)
        summary_rows.append({
            "clustering": name,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "n_clusters": len(counts),
            "tiny_clusters": tiny,
            "huge_clusters": huge,
        })

        # build cluster size df for its sheet
        size_df = pd.DataFrame({"cluster": list(counts.keys()),
                               "size": list(counts.values())})
        size_sheets[name[:31]] = size_df.sort_values("size", ascending=False)

    # write Excel
    out = args.out
    with pd.ExcelWriter(out, engine="xlsxwriter") as xl:
        pd.DataFrame(summary_rows).to_excel(xl, sheet_name="summary", index=False)
        for sheet_name, df in size_sheets.items():
            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", sheet_name)[:31]
            df.to_excel(xl, sheet_name=safe_name, index=False)

    print(f"✅ Metrics written to {out.resolve()}")


if __name__ == "__main__":
    main()
