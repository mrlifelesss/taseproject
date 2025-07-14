import pandas as pd

# 1) Load your two reports
hdb = pd.read_excel("hdbscan_search_report.xlsx")
meta = pd.read_excel("meta_search_report.xlsx")

# generate a human‐readable combo field
hdb["combo"] = hdb.apply(
    lambda r: f"pca{int(r['pca_dim'])}"
              f"_min{int(r['min_cluster_size'])}"
              f"_samp{int(r['min_samples'])}"
              f"_eps{r['eps']:.2f}",
    axis=1
)
# build combo for (k_base, hdb_min, k_meta)
meta["combo"] = meta.apply(
    lambda r: f"kbase{int(r['k_base'])}"
              f"_hdb{int(r['hdb_min'])}"
              f"_kmeta{int(r['k_meta'])}",
    axis=1
)

# 2) Keep only runs with ≤ 20 clusters
hdb_20 = hdb[(hdb.n_clusters >= 5) & (hdb.n_clusters <= 20)]
meta_20 = meta[(meta.n_meta_clusters >= 5) & (meta.n_meta_clusters <= 20)]

# 3) Sort: best silhouette first, break ties by low DBI
hdb_best = hdb_20.sort_values(
    ["silhouette", "davies_bouldin"],
    ascending=[False, True]
).head(5)

meta_best = meta_20.sort_values(
    ["silhouette_cosine", "davies_bouldin"],
    ascending=[False, True]
).head(5)

TOTAL_DOCS = 6249   # or compute dynamically if you prefer
hdb_best["pct_noise"] = hdb_best["n_noise"] / TOTAL_DOCS * 100


print("Top 5 HDBSCAN runs (≤20 clusters):")
print(hdb_best[[
    "combo","n_clusters","pct_noise",
    "silhouette","davies_bouldin"
]])

print("\nTop 5 Meta-clustering runs (≤20 clusters):")
print(meta_best[[
    "combo","n_meta_clusters",
    "silhouette_cosine","davies_bouldin"
]])
