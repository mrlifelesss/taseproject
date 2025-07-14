#!/usr/bin/env python3
"""
cluster_center_distances.py

Compute cluster distances and sample documents, including cluster metadata and event name mapping.

Outputs JSON reports with:
- cluster_id
- theme_title_hebrew
- keywords_hebrew
- common_form_types
- common_event_ids
- center: 8 closest docs (with id, distance, form_type, events, text)
- farthest: 7 farthest docs

Usage example:
python cluster_center_distances.py \
    --emb doc_vectors.parquet \
    --doc-map-json doc_cluster_map.json \
    --cluster-meta-json cluster_analysis3.json \
    --event-id-map event_id_name_map_full.json \
    --clusters CORP_INSIDER_HOLDINGS ... \
    --text-dir cleanHTM \
    --output-dir cluster_distance_reports \
    --excel gmm_sampled_and_enriched.xlsx \
    --sheet comp44
"""
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# --------------------------------------------------------------------------- #
def load_metadata(excel_path: Path, sheet: str = "comp44") -> dict[str, dict]:
    if not excel_path.is_file():
        logging.warning("Excel file %s not found – metadata will be empty.", excel_path)
        return {}
    df = pd.read_excel(excel_path, sheet_name=sheet)
    meta: dict[str, dict] = {}
    for _, row in df.iterrows():
        did = str(row.get("doc_id", row.get("document_id", ""))).strip()
        if not did:
            continue
        raw_eids = row.get("eventIds")
        if pd.isna(raw_eids):
            eids = []
        elif isinstance(raw_eids, (list, tuple)):
            eids = list(raw_eids)
        elif isinstance(raw_eids, str):
            parts = [p.strip() for p in raw_eids.split(",") if p.strip()]
            try:
                eids = [int(p) for p in parts]
            except ValueError:
                eids = []
        else:
            eids = []
        meta[did] = {"form_type": row.get("form_type"), "eventIds": eids}
    logging.info("Metadata loaded for %d docs from %s (sheet %s)", len(meta), excel_path.name, sheet)
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


def load_doc_map(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    doc_map: dict[str, list[str]] = {}
    if isinstance(data, list):
        for info in data:
            cid = info.get("cluster_id")
            docs = info.get("document_ids") or []
            doc_map[cid] = [str(d) for d in docs]
    else:
        for section in (data.get("final_categories", {}), data.get("provisional_candidate_groups", {})):
            for cid, info in section.items():
                docs = info.get("document_ids") or []
                doc_map[cid] = [str(d) for d in docs]
    logging.info("Loaded document map for %d clusters from %s", len(doc_map), path.name)
    return doc_map


def load_cluster_meta(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta_map: dict[str, dict] = {}
    if not isinstance(data, list):
        raise ValueError("Cluster metadata JSON must be a list of cluster objects.")
    for info in data:
        cid = info.get("cluster_id")
        if not cid:
            continue
        meta_map[cid] = {
            "theme_title_hebrew": info.get("theme_title_hebrew"),
            "keywords_hebrew": info.get("keywords_hebrew"),
            "common_form_types": info.get("common_form_types"),
            "common_event_ids": info.get("common_event_ids"),
        }
    logging.info("Loaded cluster metadata for %d clusters from %s", len(meta_map), path.name)
    return meta_map


def cosine_distances(X: np.ndarray, centre: np.ndarray) -> np.ndarray:
    centre = centre / np.linalg.norm(centre)
    return cdist(X, centre[None, :], metric="cosine").flatten()


def load_text(doc_id: str, text_dir: Path) -> str | None:
    file_path = text_dir / f"{doc_id}.txt"
    if file_path.is_file():
        try:
            return file_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logging.warning("Could not read %s: %s", file_path, e)
    return None

# --------------------------------------------------------------------------- #
def analyse_and_save(
    cid: str,
    docs: list[str],
    emb_lookup: dict[str, np.ndarray],
    text_dir: Path,
    out_dir: Path,
    meta_lookup: dict[str, dict],
    cluster_meta: dict,
    event_map: dict
):
    vecs, ids = [], []
    for d in docs:
        if d in emb_lookup:
            ids.append(d)
            vecs.append(emb_lookup[d])
    if not vecs:
        logging.warning("Cluster %s has no embeddings present, skipping.", cid)
        return

    X = np.vstack(vecs)
    centre_vec = X.mean(axis=0)
    dists = cosine_distances(X, centre_vec)
    order = np.argsort(dists)

    idx_center = order[:8]
    idx_farthest = order[-7:][::-1]

    def bundle(indices):
        out: list[dict] = []
        for i in indices:
            did = ids[i]
            info = meta_lookup.get(did, {})
            eids = info.get("eventIds") or []
            events = [{"id": int(e), "name": event_map.get(str(e))} for e in eids]
            out.append({
                "doc_id": did,
                "cosine_distance": float(dists[i]),
                "form_type": info.get("form_type"),
                "events": events,
                "text": load_text(did, text_dir),
            })
        return out

    report = {
        "cluster_id": cid,
        **cluster_meta,
        "center": bundle(idx_center),
        "farthest": bundle(idx_farthest),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cid}_distance_report.json"
    out_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logging.info("→ Saved %s", out_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, type=Path,
                    help="Embeddings parquet file.")
    ap.add_argument("--doc-map-json", required=True, type=Path,
                    help="JSON mapping clusters to document_ids.")
    ap.add_argument("--cluster-meta-json", required=True, type=Path,
                    help="JSON list of cluster metadata (analysis3.json).")
    ap.add_argument("--event-id-map", required=True, type=Path,
                    help="JSON map of event ID to name.")
    ap.add_argument("--clusters", nargs="+", required=True,
                    help="List of cluster IDs to process.")
    ap.add_argument("--text-dir", required=True, type=Path,
                    help="Directory of cleaned text files.")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Output directory for reports.")
    ap.add_argument("--excel", required=True, type=Path,
                    help="Excel file for doc metadata.")
    ap.add_argument("--sheet", default="comp44",
                    help="Worksheet name (default comp44). ")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    emb_lookup = load_embeddings(args.emb)
    doc_map = load_doc_map(args.doc_map_json)
    cluster_meta = load_cluster_meta(args.cluster_meta_json)
    event_map = json.loads(args.event_id_map.read_text(encoding="utf-8"))
    meta_lookup = load_metadata(args.excel, args.sheet)

    for cid in args.clusters:
        docs = doc_map.get(cid, [])
        if len(docs) < 10:
            logging.warning("Skipping cluster %s: only %d docs (<15), skipping.", cid, len(docs))
            continue
        meta = cluster_meta.get(cid, {})
        analyse_and_save(
            cid,
            docs,
            emb_lookup,
            args.text_dir,
            args.output_dir,
            meta_lookup,
            meta,
            event_map
        )

if __name__ == "__main__":
    main()
