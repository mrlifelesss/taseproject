# build_doc_json.py
#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd

EXCLUDE_CLUSTERS = {41, 42, 43, 40, 37, 35, 33, 31, 23}
POSITION_SAMPLES = {
    "center": 2,
    "middle": 3,
    "edge": 4,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build remaining JSON records based on an existing sample or by sampling positions from clusters"
    )
    parser.add_argument("--excel-path", required=True,
                        help="Path to gmm_sampled_and_enriched.xlsx")
    parser.add_argument("--sheet-name", default=None,
                        help="Excel sheet name (default: first sheet)")
    parser.add_argument("--clean-dir", required=True,
                        help="Directory containing cleaned .txt files named by doc_id")
    parser.add_argument("--sample-file", default=None,
                        help="Path to an existing sampled JSON file (skips sampling)")
    parser.add_argument("--sampled-output", default="sampled_docs.json",
                        help="Output JSON file for sampled records")
    parser.add_argument("--remaining-output", default="remaining_docs.json",
                        help="Output JSON file for remaining records")
    parser.add_argument("--cluster-col", default=None,
                        help="Column name for cluster (defaults to first 'cluster_*')")
    parser.add_argument("--position-col", default="position_category",
                        help="Column name for position category (center/middle/edge)")
    parser.add_argument("--form-type-col", default="form_type",
                        help="Column name for form type")
    parser.add_argument("--eventids-col", default="eventIds",
                        help="Column name for event IDs")
    parser.add_argument("--distance-col", default="distance_to_center",
                        help="Column name for distance_to_center")
    args = parser.parse_args()

    # Load DataFrame
    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)

    # Determine cluster column
    if args.cluster_col:
        cluster_col = args.cluster_col
    else:
        clusters = [c for c in df.columns if c.lower().startswith("cluster")]
        if not clusters:
            raise ValueError("No column starting with 'cluster' found in Excel")
        cluster_col = clusters[0]

    sampled_ids = set()
    counter = 1

    # Load existing sample if provided
    if args.sample_file:
        with open(args.sample_file, 'r', encoding='utf-8') as sf:
            sampled_records = json.load(sf)
        # extract sampled IDs and next counter
        for rec in sampled_records:
            sampled_ids.add(int(rec.get('doc_id', rec.get('doc_id'))))
        # determine counter start from max count
        max_count = max((rec.get('count', 0) for rec in sampled_records), default=0)
        counter = max_count + 1
        # write sample unchanged
        with open(args.sampled_output, 'w', encoding='utf-8') as fp:
            json.dump(sampled_records, fp, ensure_ascii=False, indent=2)
        print(f"Loaded {len(sampled_records)} sampled records from {args.sample_file}")
    else:
        # perform sampling
        sampled_records = []
        for cluster_id, group in df.groupby(cluster_col):
            if int(cluster_id) in EXCLUDE_CLUSTERS:
                continue

            parts = []
            for pos_label, n in POSITION_SAMPLES.items():
                subset = group[group[args.position_col].str.lower() == pos_label]
                if subset.empty:
                    continue
                count = min(n, len(subset))
                parts.append(subset.sample(n=count, random_state=42))
            if not parts:
                continue
            sample_df = pd.concat(parts)

            for _, row in sample_df.iterrows():
                doc_id = int(row['doc_id'])
                sampled_ids.add(doc_id)
                rec = {
                    'count': counter,
                    'doc_id': doc_id,
                    'form_type': row.get(args.form_type_col, ''),
                    'eventIds': row.get(args.eventids_col, []),
                    'cluster': int(cluster_id),
                    'position_category': row.get(args.position_col, ''),
                    'distance_to_center': row.get(args.distance_col, None),
                    'text': ''
                }
                txt_path = os.path.join(args.clean_dir, f"{doc_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        rec['text'] = f.read().strip()
                sampled_records.append(rec)
                counter += 1
        # write sample
        with open(args.sampled_output, 'w', encoding='utf-8') as fp:
            json.dump(sampled_records, fp, ensure_ascii=False, indent=2)
        print(f"Wrote {len(sampled_records)} sampled records to {args.sampled_output}")

    # Build remaining records
    remaining_records = []
    for _, row in df.iterrows():
        doc_id = int(row['doc_id'])
        if doc_id in sampled_ids:
            continue
        rec = {
            'count': counter,
            'doc_id': doc_id,
            'form_type': row.get(args.form_type_col, ''),
            'eventIds': row.get(args.eventids_col, []),
            'cluster': row.get(cluster_col),
            'position_category': row.get(args.position_col, ''),
            'distance_to_center': row.get(args.distance_col, None),
            'text': ''
        }
        txt_path = os.path.join(args.clean_dir, f"{doc_id}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                rec['text'] = f.read().strip()
        remaining_records.append(rec)
        counter += 1

    # write remaining JSON
    with open(args.remaining_output, 'w', encoding='utf-8') as fp:
        json.dump(remaining_records, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(remaining_records)} remaining records to {args.remaining_output}")


if __name__ == '__main__':
    main()
