#!/usr/bin/env python3

"""Combined embedding and clustering pipeline.

Pipeline steps:
1. clean_text_test.py - Clean raw HTML files into text
2. extract_and_chunk2.py - Extract and semantically chunk HTML from DynamoDB/S3
3. pool_chunks_to_docs.py - Pool chunk embeddings into document-level vectors
4. embed_chunks_local.py - Run embedding with HuggingFace models
5. cluster_gmm.py - Cluster documents using GMM
6. sample_cluste.py - Sample representative docs from each cluster
"""

import subprocess
import threading
from pathlib import Path
import sys

def run_script(script_path, args):
    command = [sys.executable, str(script_path)] + args
    print(f"\n▶ Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Script failed: {script_path}")
        print(f"Exit Code: {e.returncode}")
        print(f"Command: {' '.join(command)}")
        print("------- STDOUT -------")
        print(e.stdout)
        print("------- STDERR -------")
        print(e.stderr)
        raise

def main():
    scripts_dir = Path(__file__).parent

    threads = []

    # 2. Extract and chunk documents
    threads.append(threading.Thread(target=run_script, args=(
        str(scripts_dir / "extract_and_chunk2.py"),
        ["--out-dir", "cleaned_text"],
    )))

    # Start HTML preprocessing and chunking in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 3. Embed chunked documents
    run_script(str(scripts_dir / "embed_chunks_local.py"), [
        "--models", "BAAI/bge-m3",
        "--src", "cleaned_text",
        "--dst", "embeddings_dual",
    ])

    # 4. Pool embeddings to doc-level
    run_script(str(scripts_dir / "pool_chunks_to_docs.py"), [
        "--src", "embeddings_dual/BAAI_bge-m3",
        "--dst", "doc_vectors_bge.parquet",
        "--method", "mean",
    ])

    # 5. Cluster documents
    run_script(str(scripts_dir / "cluster_gmm.py"), [
        "--input", "doc_vectors_bge.parquet",
        "--output", "gmm_clusters_comp20_covtied_pca50.csv",
    ])

    # 6. Sample cluster regions
    run_script(str(scripts_dir / "sample_cluste.py"), [
        "--doc_vectors", "doc_vectors_bge.parquet",
        "--cluster_csv", "gmm_clusters_comp20_covtied_pca50.csv",
        "--output", "sampled_examples.csv",
    ])

if __name__ == "__main__":
    main()
