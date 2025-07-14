#!/usr/bin/env python3

import argparse
import json
import re
import os
import pandas as pd
import google.generativeai as genai
import time

def load_samples(excel_path: str, cluster_suffix: str):
    # derive sheet name (e.g. suffix=38 → sheet "comp38")
    sheet_name = f"comp{cluster_suffix}"
    # read only that sheet
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # derive the column name (e.g. "cluster_38")
    cluster_col = f"cluster_{cluster_suffix}"
    if cluster_col not in df.columns:
        raise ValueError(
            f"Column '{cluster_col}' not found in sheet '{sheet_name}'. "
            f"Available columns: {', '.join(df.columns)}"
        )
    # group *all* rows by their cluster ID
    return df.groupby(cluster_col)


def fetch_local_excerpt(clean_dir: str, doc_id: int, max_chars: int = 500) -> str:
    """
    Look for <clean_dir>/<doc_id>.txt (or .html/.htm), read it and return
    the first max_chars characters.
    """
    base = os.path.join(clean_dir, str(doc_id))
    for ext in (".txt", ".html", ".htm"):
        path = base + ext
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:max_chars]
    return ""

def build_prompt(cluster_id: int, excerpts: list[str], template: str) -> str:
    block = ""
    for i, excerpt in enumerate(excerpts, start=1):
        block += f"---\nExcerpt {i}:\n```\n{excerpt}\n```\n"
    prompt = template.replace("[Cluster_ID_Placeholder]", str(cluster_id))
    return prompt.replace("{excerpts_block}", block)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clusters with Gemini LLM"
    )
    parser.add_argument("--excel-path", required=True,
                        help="Path to gmm_sampled_and_enriched.xlsx")
    parser.add_argument("--cluster_suffix", required=True,
                        help="Numeric suffix of your clustering, e.g. 38, 44, or 50")
    parser.add_argument("--clean-dir", required=True,
                        help="Directory where cleaned files live (e.g. cleanHTM)")
    parser.add_argument("--model-name", default="gemini-1.5-flash-latest",
                        help="Gemini model name")
    parser.add_argument("--gemini-api-key", default= os.getenv("gemini_key"),
                        help="Gemini API key")
    parser.add_argument("--num-samples", type=int, default=12,
                        help="Samples per cluster")
    parser.add_argument("--output-path", default="llm_evaluations.json",
                        help="Output JSON path")
    args = parser.parse_args()


    # Gemini setup
    genai.configure(api_key= "gemini_key")
    model = genai.GenerativeModel(args.model_name)

    # Load clusters
    clusters = load_samples(args.excel_path, args.cluster_suffix)

    # Prompt template (from thing.md)
    prompt_template = """
You are an expert document analyst specializing in financial announcements.
Your task is to analyze a sample of texts from a single pre-defined cluster of company announcements.

**Instructions:**
Based on the provided document excerpts below, which all belong to the same cluster:

1.  **Cluster Theme/Title:** What is the single, primary common theme or subject matter of this group of texts? Provide a concise title (3-7 words, in Hebrew) for this theme.
2.  **Key Differentiating Keywords/Phrases:** List 5-7 distinct keywords or key phrases (in Hebrew, can include relevant English terms if they appear consistently) that best characterize this group and would help differentiate its documents from other types of announcements.
3.  **Coherence for Summarization Prompting:** Briefly (1-2 sentences, in Hebrew or English as you see fit) explain if these documents seem thematically coherent enough to be effectively summarized using a single, tailored summarization prompt. Note any significant diversity within the samples that might challenge this.
4.  **Confidence Score (1-5):** On a scale of 1 (Not coherent, diverse topics) to 5 (Very coherent, single clear topic), how confident are you that a single summarization prompt would be suitable for all documents in this cluster, based on these samples?

**Document Excerpts from Cluster [Cluster_ID_Placeholder]:**
{excerpts_block}

**Your Structured Output (Provide only this JSON structure):**
```json
{
  "cluster_id": "[Cluster_ID_Placeholder]",
  "theme_title_hebrew": "string",
  "keywords_hebrew": ["string", "string", ...],
  "coherence_explanation": "string",
  "prompt_suitability_confidence": "integer (1-5)"
}```"""

    results = []

    for cluster_id, group in clusters:
        report_ids = (group['doc_id']
                      .dropna()
                      .astype(int)
                      .unique()[: args.num_samples])

        excerpts = []
        for rid in report_ids:
            text = fetch_local_excerpt(args.clean_dir, rid, max_chars=500)
            if not text:
                 print(f"   No cleaned file for {rid}, skipping")
                 continue
            excerpts.append(text)
            if len(excerpts) >= args.num_samples:
                break

        if len(excerpts) < 2:
            print(f"Skipping cluster {cluster_id}: insufficient excerpts")
            continue

        prompt = build_prompt(cluster_id, excerpts, prompt_template)
        response = model.generate_content(prompt)
        time.sleep(4)
        raw = response.text.strip()
        
        # remove any leading ```json and trailing ```
        # 1) Strip leading ```json (and optional newline)
        json_str = re.sub(r"^```json\s*\n?", "", raw)
        # 2) Strip trailing ``` if present
        json_str = re.sub(r"\n?```$", "", json_str)
        # finally strip whitespace
        json_str = json_str.strip()

# ---- ESCAPE unescaped quotes between Hebrew letters ----
        # Unicode range \u0590–\u05FF covers Hebrew block
        json_str = re.sub(
           r'(?<=[\u0590-\u05FF])"(?=[\u0590-\u05FF])',            r'\\"',
          json_str
          )

        
        
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            data = {
            "cluster_id": cluster_id,
            "error": f"JSON decode error: {e}",
            "raw_response": raw
                }

        results.append(data)
        print(f"Cluster {cluster_id} evaluated.")

    # Write output
    with open(args.output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()