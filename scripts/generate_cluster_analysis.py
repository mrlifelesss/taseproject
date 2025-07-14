#!/usr/bin/env python3
"""
analyze_clusters_with_file_api.py

Generate cluster excerpt JSON files and analyze each via Gemini File API,
producing structured JSON outputs per cluster.
"""
import argparse
import json
import sys
import logging
from pathlib import Path
from google import genai
import os


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_cluster_mapping(json_path: Path) -> dict[str, list[str]]:
    """
    Load cluster→[doc_id,…] from either:
      - old format with "final_categories"/"provisional_candidate_groups" and "document_ids"
      - new top-level format with "doc_ids" or "document_ids" per cluster
    """
    raw = json_path.read_bytes().decode('utf-8-sig', errors='replace')
    data = json.loads(raw)
    clusters: dict[str, list[str]] = {}

    if "final_categories" in data or "provisional_candidate_groups" in data:
        # legacy two-section format
        for section in ("final_categories", "provisional_candidate_groups"):
            for cid, info in data.get(section, {}).items():
                ids = info.get("document_ids", [])
                clusters[cid] = [str(d) for d in ids]
    else:
        # top-level mapping: each key is a cluster
        for cid, info in data.items():
            # handle both "doc_ids" and "document_ids"
            ids = info.get("doc_ids") or info.get("document_ids") or []
            clusters[cid] = [str(d) for d in ids]

    logging.info("Loaded %d clusters from %s", len(clusters), json_path.name)
    return clusters


def sample_texts(doc_ids: list[str], text_dir: Path, sample_size: int) -> list[str]:
    """Read up to sample_size text files for the given doc_ids."""
    excerpts = []
    for doc_id in doc_ids[:sample_size]:
        txt_path = text_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            logging.warning(f"Missing text for doc_id {doc_id} at {txt_path}")
            continue
        text = txt_path.read_text(encoding='utf-8').strip()
        excerpts.append(text)
    return excerpts


def write_excerpt_file_txt(cid: str, excerpts: list[str], output_dir: Path) -> Path:
    """Write a structured TXT file with cluster_id and excerpts."""
    lines = [f"cluster_id: {cid}", "excerpts:"]
    for ex in excerpts:
        lines.append("  - |")
        for l in ex.splitlines():
            lines.append(f"    {l}")
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    out_path = output_dir / f"cluster_{cid}.txt"
    out_path.write_text(content, encoding='utf-8')
    logging.info(f"Wrote structured TXT: {out_path} ({len(excerpts)} excerpts)")
    return out_path


def upload_file_to_gemini(file_path: Path, client):
    """Upload file to Gemini File API and return file_id."""
    logging.info(f"[MOCK] Uploading {file_path} to Gemini File API")
    file = client.files.upload(file= file_path)
    return file #this  is a myfile to to  puit into the modal call in the form of [prompt , myfile]


def build_file_prompt(cid: str) -> str:
    """Construct the prompt for analysis using the file reference."""
    return f"""
You are an expert document analyst specializing in financial announcements.
Your task is to analyze a sample of texts from a single pre-defined cluster of company announcements.

**Instructions:**
Based on the attached document excerpts (provided via file reference), which all belong to the same cluster:

1.  **Cluster Theme/Title:** What is the single, primary common theme or subject matter of this group of texts? Provide a concise title (3-7 words, in Hebrew) for this theme.
2.  **Key Differentiating Keywords/Phrases:** List 5-7 distinct keywords or key phrases (in Hebrew, can include relevant English terms if they appear consistently) that best characterize this group and would help differentiate its documents from other types of announcements.
3.  **Coherence for Summarization Prompting:** Briefly (1-2 sentences, in Hebrew or English as you see fit) explain if these documents seem thematically coherent enough to be effectively summarized using a single, tailored summarization prompt. Note any significant diversity within the samples that might challenge this.
4.  **Confidence Score (1-5):** On a scale of 1 (Not coherent, diverse topics) to 5 (Very coherent, single clear topic), how confident are you that a single summarization prompt would be suitable for all documents in this cluster, based on these samples?

Please respond strictly with this JSON structure:
```json
{{
  "cluster_id": "{cid}",
  "theme_title_hebrew": "string",
  "keywords_hebrew": ["string", ...],
  "common_form_types": ["string", ...],
  "common_event_ids": ["int", ...],
  "coherence_explanation": "string",
  "prompt_suitability_confidence": integer
}}
```"""


def analyze_file_with_gemini(file_id: str, prompt: str , client) -> dict:
    """Generate analysis via Gemini using file_id and prompt."""
    logging.info(f" Generating content for file_id {file_id}")
    response = client.models.generate_content(
    model="gemini-2.0-flash", contents=[prompt, file_id]
    )
    raw = response.text
    clean = clean_json_fence(raw)
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode failed: {e}")
        logging.error("Response text was:\n" + clean)
        # you can choose to re-raise or return a safe default
        return {
            "cluster_id": file_id,
            "theme_title_hebrew": "",
            "keywords_hebrew": [],
            "coherence_explanation": "",
            "prompt_suitability_confidence": 0
        }
def clean_json_fence(raw: str) -> str:
    """If the model wraps its JSON in ```json ... ```, strip those fences."""
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        # drop first fence line
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        # drop closing fence line
        lines = lines[:-1]
    return "\n".join(lines).strip()

def main():
    parser = argparse.ArgumentParser(
        description="Extract cluster excerpts and analyze via Gemini File API"
    )
    parser.add_argument("--cluster-json", type=Path, required=True,
                        help="Path to clustering JSON file")
    parser.add_argument("--text-dir", type=Path, required=True,
                        help="Directory of document text files named <doc_id>.txt")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write cluster excerpt JSON files")
    parser.add_argument("--analysis-output", type=Path, required=True,
                        help="Path to write aggregated analysis results as JSON")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="Minimum number of documents per cluster to include")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of text samples to include per cluster")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    print(api_key)
    client = genai.Client(api_key= api_key)
    
    
    clusters = load_cluster_mapping(args.cluster_json)
    # filter out noise and missing metadata clusters
    ignore_clusters = {"UNCLASSIFIED_NOISE", "MISSING_ID_METADATA", "CORP_GOVERNANCE_HOLDINGS"}
    valid_clusters = {cid: ids for cid, ids in clusters.items() if cid not in ignore_clusters}
    # filter by size
    #valid_clusters = {cid: ids for cid, ids in filtered.items() if len(ids) >= args.min_cluster_size}
    #if not valid_clusters:
    #    logging.error(f"No clusters with >= {args.min_cluster_size} docs")
    #    sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for cid, doc_ids in valid_clusters.items():
        excerpts = sample_texts(doc_ids, args.text_dir, args.sample_size)
        if not excerpts:
            logging.warning(f"No excerpts for cluster {cid}, skipping")
            continue
        # write intermediate TXT
        excerpt_file = write_excerpt_file_txt(cid, excerpts, args.output_dir)
        # upload and analyze
        file = upload_file_to_gemini(excerpt_file, client)
        prompt = build_file_prompt(cid)
        analysis = analyze_file_with_gemini(file, prompt , client)
        analysis["cluster_id"] = cid
        results.append(analysis)

    # write aggregated analysis
    with open(args.analysis_output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Wrote analysis for {len(results)} clusters to {args.analysis_output}")


if __name__ == "__main__":
    main()
