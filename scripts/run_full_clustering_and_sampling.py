#!/usr/bin/env python3
"""Full pipeline: sample 12 docs for EACH cluster (center, middle, edge categories) and enrich metadata"""
import json
import concurrent.futures
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
import boto3
import pyarrow.parquet as pq
from boto3.dynamodb.conditions import Key
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------- CONFIG --------------------
INPUT_PARQUET = "doc_vectors.parquet"
OUTPUT_PREFIX = "gmm_clusters_covtied_pca50"
COMPONENTS = [50, 48, 44, 38]
N_DOCS_EACH = 4  # samples per category
DYNAMODB_TABLE = "CompanyDisclosuresHebrew"
GSI_NAME = "reportId-index"
EMBED_COL = "embedding"
OUT_XLSX = "gmm_sampled_and_enriched.xlsx"
# ------------------------------------------------

# Setup DynamoDB client/table
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMODB_TABLE)

# Utility to normalize embeddings
def l2_norm(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

# Unwrap DynamoDB attribute-types recursively
def unwrap_attr(v):
    if isinstance(v, dict) and len(v) == 1:
        t, val = next(iter(v.items()))
        if t == "N":  # number
            try: return int(val)
            except: 
                try: return float(val)
                except: return val
        if t in ("S", "BOOL"):
            return val
        if t == "M":
            return {k: unwrap_attr(x) for k, x in val.items()}
        if t == "L":
            return [unwrap_attr(x) for x in val]
    return v

# Extract all eventIds from nested dict/list
def extract_event_ids(obj):
    ids = []
    def rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "eventId":
                    v_native = unwrap_attr(v)
                    try:
                        ids.append(int(v_native))
                    except:
                        pass
                else:
                    rec(v)
        elif isinstance(o, list):
            for item in o:
                rec(item)
    rec(obj)
    return list(dict.fromkeys(ids))

# Fetch metadata from DynamoDB for a given reportId
def fetch_metadata(report_id):
    try:
        resp = table.query(
            IndexName=GSI_NAME,
            KeyConditionExpression=Key("reportId").eq(str(report_id))
        )
        if not resp.get("Items"):
            return {"reportId": report_id, "form_type": None, "eventIds": None}
        item = resp["Items"][0]
        form_type = item.get("form_type")
        events_raw = item.get("events", {})
        # If stringified JSON, parse
        if isinstance(events_raw, str):
            try:
                events_raw = json.loads(events_raw)
            except:
                events_raw = {}
        events_unwrapped = unwrap_attr(events_raw)
        ids = extract_event_ids(events_unwrapped)
        event_ids_str = ",".join(map(str, ids)) if ids else None
        return {"reportId": report_id, "form_type": form_type, "eventIds": event_ids_str}
    except Exception as e:
        return {"reportId": report_id, "form_type": None, "eventIds": None, "error": str(e)}

# Enrich a list of reportIds with metadata via multithreading
def enrich_report_ids(ids):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        metas = list(executor.map(fetch_metadata, ids))
    return pd.DataFrame(metas)

# Sample docs with position labels for a single cluster
def sample_docs_with_category(df, cluster_col, cluster_id):
    subset = df[df[cluster_col] == cluster_id].copy()
    embeddings = np.stack(subset[EMBED_COL].values)
    centroid = embeddings.mean(axis=0)
    distances = euclidean_distances(embeddings, [centroid]).ravel()
    subset["distance_to_center"] = distances
    subset = subset.sort_values("distance_to_center")
    # pick samples
    center = subset.head(N_DOCS_EACH).copy()
    center["position_category"] = "center"
    edge = subset.tail(N_DOCS_EACH).copy()
    edge["position_category"] = "edge"
    mid_start = max((len(subset)//2) - (N_DOCS_EACH//2), 0)
    middle = subset.iloc[mid_start: mid_start + N_DOCS_EACH].copy()
    middle["position_category"] = "middle"
    return pd.concat([center, middle, edge])

def plot_clusters_2d(data, labels, title, filename):
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=2)
    reduced = pca_2d.fit_transform(data)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='tab20', legend=False, s=10)
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Load embeddings
    table_data = pq.read_table(INPUT_PARQUET)
    df = table_data.to_pandas()
    if "vector" not in df.columns or "doc_id" not in df.columns:
        raise ValueError("Input parquet must have 'doc_id' and 'vector' columns'")
    # Normalize and reduce dimension
    embeddings = np.stack(df["vector"].apply(np.array).values)
    embeddings = l2_norm(embeddings)
    reduced = PCA(n_components=50, random_state=42).fit_transform(embeddings)
    df[EMBED_COL] = list(reduced)

    all_samples = []

    # For each clustering config
    for n in COMPONENTS:
        # GMM clustering
        labels = GaussianMixture(
            n_components=n, covariance_type="tied", random_state=42
        ).fit_predict(reduced)
        col_name = f"cluster_{n}"
        df[col_name] = labels
        # Save clusters CSV
        pd.DataFrame({
            "doc_id": df["doc_id"], 
            "cluster": labels
        }).to_csv(f"{OUTPUT_PREFIX}_comp{n}.csv", index=False)

        # For each cluster in this config
        unique_clusters = sorted(df[col_name].unique())
        for cid in unique_clusters:
            sampled = sample_docs_with_category(df, col_name, cid)
            meta = enrich_report_ids(sampled["doc_id"].tolist())
            merged = sampled.merge(meta, left_on="doc_id", right_on="reportId", how="left")
            merged["config"] = f"comp{n}"
            merged["cluster_id"] = cid
            all_samples.append(merged)
    # Combine and write to Excel: each config sheet
    all_df = pd.concat(all_samples, ignore_index=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        for n in COMPONENTS:
            sheet = f"comp{n}"
            subset = all_df[all_df["config"] == f"comp{n}"]
            subset.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Completed. Output saved to {OUT_XLSX}")

if __name__ == "__main__":
    main()