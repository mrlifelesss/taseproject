#!/usr/bin/env python
# flatten_clustering3.py
"""
Turn the nested structure in clustering3.json into a flat list that
make_cluster_distance_samples.py already understands.
"""

import json
from pathlib import Path

SRC = Path("inputs/clustering3.json")          # <-- your uploaded file
DST = Path("clustering3_flat.json")     # will be created/overwritten

EXCLUDE = {                             # clusters you don’t want
    "UNCLASSIFIED_NOISE",
    "MISSING_ID_METADATA",
    "CORP_GOVERNANCE_HOLDINGS",
    "CORP_GOVERNANCE_OFFICER_CHANGES",
    "CORP_MEETINGS_BONDHOLDER_SHAREHOLDER",
}

def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))

    # merge both top-level blocks: final_categories + provisional_candidate_groups
    flat: list[dict] = []
    for outer in ("final_categories", "provisional_candidate_groups"):
        for cid, info in data.get(outer, {}).items():
            if cid in EXCLUDE:
                continue
            ids = [str(x) for x in info.get("document_ids", [])]  # cast → str
            if not ids:                                           # skip empty clusters
                continue
            flat.append({"cluster_id": cid, "document_ids": ids})

    # (optional) sort clusters by size, largest first
    flat.sort(key=lambda d: len(d["document_ids"]), reverse=True)

    DST.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✔ wrote {DST}  ({len(flat)} clusters)")

if __name__ == "__main__":
    main()
