import pandas as pd
df = pd.read_parquet("doc_vectors_BAAI.parquet")
df.to_csv("doc_vectors2.csv", index=False)
