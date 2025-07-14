#!/usr/bin/env python3
"""
cluster_center_distances.py

Compute cluster centres and distance rankings, then save the
10 closest / middle / farthest docs *with their cleaned text*.

Usage example:
python cluster_center_distances.py \
    --emb doc_vectors.parquet \
    --json code.json \
    --clusters CORP_GOVERNANCE_OFFICER_CHANGES \
               CORP_MEETINGS_BONDHOLDER_SHAREHOLDER \
    --text-dir cleanHTM \
    --output-dir cluster_distance_reports
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd 
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# --------------------------------------------------------------------------- #
def load_metadata(excel_path: Path, sheet: str = "comp44") -> dict[str, dict]:
    """
    Returns {doc_id: {"form_type": str | None, "eventIds": list[int] | None}}
    """
    if not excel_path.is_file():
        logging.warning("Excel file %s not found – metadata will be empty.", excel_path)
        return {}

    df = pd.read_excel(excel_path, sheet_name=sheet)
    meta = {}
    for _, row in df.iterrows():
        did = str(row.get("doc_id", row.get("document_id", ""))).strip()
        if not did:
            continue
        raw_eids = row.get("eventIds")

        if pd.isna(raw_eids):             # blank / NaN → None
            eids = None
        elif isinstance(raw_eids, (list, tuple)):
            eids = list(raw_eids)
        elif isinstance(raw_eids, str):
            # split by comma, ignore empty parts, cast to int if possible
            parts = [p.strip() for p in raw_eids.split(",") if p.strip()]
            try:
                eids = [int(p) for p in parts]
            except ValueError:
                # keep as strings if any part is non-numeric
                eids = parts
        else:
            eids = None

        meta[did] = {
            "form_type": row.get("form_type"),
            "eventIds": eids,
        }
    logging.info("Metadata loaded for %d docs from %s (sheet %s)",
                 len(meta), excel_path.name, sheet)
    return meta

def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    if "vector" in df.columns:
        vecs = np.vstack(df["vector"].to_list())
    else:
        cols = [c for c in df.columns if c != "doc_id"]
        vecs = df[cols].values
    ids = df["doc_id"].astype(str).tolist()
    return dict(zip(ids, vecs))


def load_mapping(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clusters: dict[str, list[str]] = {}

    def ingest(section: dict):
        for cid, info in section.items():
            clusters[cid] = [str(d) for d in info.get("document_ids", [])]

    ingest(data.get("final_categories", {}))
    ingest(data.get("provisional_candidate_groups", {}))
    return clusters


def cosine_distances(X: np.ndarray, centre: np.ndarray) -> np.ndarray:
    centre = centre / np.linalg.norm(centre)
    return cdist(X, centre[None, :], metric="cosine").flatten()


def pick_indices(distances: np.ndarray, k: int = 10):
    order = np.argsort(distances)
    closest = order[:k]
    farthest = order[-k:][::-1]
    mid_start = max(0, len(distances) // 2 - k // 2)
    middle = order[mid_start:mid_start + k]
    return closest, middle, farthest


def load_text(doc_id: str, text_dir: Path) -> str | None:
    file_path = text_dir / f"{doc_id}.txt"
    if file_path.is_file():
        try:
            return file_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logging.warning("Could not read %s: %s", file_path, e)
    return None


# --------------------------------------------------------------------------- #
def analyse_and_save(cid: str,
                     doc_ids: list[str],
                     emb_lookup: dict[str, np.ndarray],
                     text_dir: Path,
                     out_dir: Path,
                      meta_lookup: dict[str, dict],
                     k: int = 10):
    # Filter to docs that have embeddings
    vecs, ids = [], []
    for d in doc_ids:
        if d in emb_lookup:
            ids.append(d)
            vecs.append(emb_lookup[d])
    if not vecs:
        logging.warning("Cluster %s has no embeddings present.", cid)
        return

    X = np.vstack(vecs)
    centre = X.mean(axis=0)
    dists = cosine_distances(X, centre)

    idx_closest, idx_middle, idx_farthest = pick_indices(dists, k=k)

    def bundle(indices):
        out = []
        for i in indices:
            did = ids[i]
            meta = meta_lookup.get(did, {})
            out.append({
                "doc_id": did,
                "cosine_distance": float(dists[i]),
                "form_type": meta.get("form_type"),
                "eventIds": meta.get("eventIds"),
                "text": load_text(did, text_dir),
            })
        return out

    cluster_report = {
        "cluster_id": cid,
        "centre": centre.tolist(),        # remove if not needed
        "closest": bundle(idx_closest),
        "middle": bundle(idx_middle),
        "farthest": bundle(idx_farthest),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cid}_distance_report.json"
    out_file.write_text(json.dumps(cluster_report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    logging.info("→ Saved %s", out_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--clusters", nargs="+", required=True,
                    help="Cluster IDs to process")
    ap.add_argument("--text-dir", required=True, type=Path,
                    help="Directory with <doc_id>.txt cleaned files")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write JSON reports")
    ap.add_argument("--excel", required=True, type=Path,
                    help="gmm_sampled_and_enriched.xlsx")
    ap.add_argument("--sheet", default="comp44",
                    help="Worksheet name inside the Excel file (default: comp44)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    emb_lookup = load_embeddings(args.emb)
    mapping = load_mapping(args.json)
    meta_lookup = load_metadata(args.excel, args.sheet)
    for cid in args.clusters:
        analyse_and_save(
            cid,
            mapping.get(cid, []),
            emb_lookup,
            args.text_dir,
            args.output_dir,
            meta_lookup,
            k=10
        )


if __name__ == "__main__":
    main()
