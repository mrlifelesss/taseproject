
"""ipol_gemini_pair.py
---------------------------------
Iterative Prompt Optimisation Loop (IPOL)
Gemini implementation that ingests **two parallel JSON files**:

    • --model_a  path/to/flash.json   (treated as Model A summaries)
    • --model_b  path/to/regular.json (treated as Model B summaries)
    • --original path/to/originals.json (optional; provides full source_text)

Each input file must be a list of records containing at least:
    { "doc_id": <int|str>, "summary": "...", ... }

For every `doc_id` present in Model A, the script looks up the matching
record in Model B (if any) and builds the structure expected by the
Iterative Prompt Optimisation Loop:

    {
      "doc_id": 123,
      "source_text": "<full text or empty>",
      "current_prompt": "",
      "summary_model_A": "...",
      "summary_model_B": "..."
    }

The record is then sent to a **teacher** Gemini model (default:
gemini‑1.5‑pro‑latest), which returns a JSON object holding:

    * gold_standard_summary
    * evaluation_scores for both models
    * verdict (winner + rationale)
    * optimized_prompt

Each result is written to --output/result_<doc_id>.json.

Requirements
------------
pip install google-generativeai tqdm

Environment:
export GOOGLE_API_KEY="YOUR_KEY"

Example
-------
python ipol_gemini_pair.py \
    --model_a gemini_responses_candidate_credit_rating_flash.json \
    --model_b gemini_responses_candidate_credit_rating.json \
    --output out/

"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import google.generativeai as genai
from tqdm import tqdm

# ------------------ Configuration ----------------------------------------

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gemini-1.5-pro-latest")
SYSTEM_PROMPT = """You are a meticulous AI Analyst and Prompt Engineer.
Non‑negotiable rules:
1. Sole Source of Truth – rely exclusively on the provided document.
2. Complete Processing – read the *entire* document.
3. If information is insufficient, report failure instead of hallucinating.
"""

META_PROMPT_TEMPLATE = """Subject: Automated Comparative Analysis and Iterative Prompt Refinement

You will receive a JSON object with the following keys:
  • doc_id
  • source_text
  • current_prompt
  • summary_model_A
  • summary_model_B

## Execution Flow
1️⃣ *Generate Gold‑Standard Summary* using `current_prompt` + `source_text`.  
2️⃣ *Evaluate* all three summaries across five categories  
    (accuracy, completeness, clarity, context, adherence) → 1‑10 per category.  
3️⃣ *Judgement* → declare which external summary is better & give one‑sentence rationale.  
4️⃣ *Prompt Optimisation* → craft `optimized_prompt` (v2) that mitigates observed faults.

## Output JSON Schema
{{
  "doc_id": "...",
  "analysis": {{
    "gold_standard_summary": "...",
    "evaluation_scores": {{
      "model_A": {{"accuracy":0,"completeness":0,"clarity":0,"context":0,"adherence":0}},
      "model_B": {{"accuracy":0,"completeness":0,"clarity":0,"context":0,"adherence":0}}
    }},
    "verdict": {{
      "winner": "Model A|Model B|Tie",
      "rationale": "..."
    }}
  }},
  "optimized_prompt": "..."
}}
Respond *only* with a valid JSON object conforming exactly to the schema above. 
"""

# ------------------ Helper functions -------------------------------------

def _messages_to_parts(messages: List[Dict[str, str]]) -> List[Dict[str, List[str]]]:
    """Convert list[{{role,content}}] → Gemini 'contents' format."""
    return [{"role": m["role"], "parts": [m["content"]]} for m in messages]

def call_gemini_chat(model_name: str, messages: List[Dict[str, str]],
                     temperature: float = 0.0, max_tokens: int = 4096) -> str:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        _messages_to_parts(messages),
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
    )
    return response.text.strip()

def run_iteration(doc: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": META_PROMPT_TEMPLATE},
        {"role": "assistant", "content": "Awaiting JSON."},
        {"role": "user",      "content": json.dumps(doc, ensure_ascii=False)}
    ]
    raw = call_gemini_chat(TEACHER_MODEL, messages)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON for doc {doc['doc_id']}") from e

# ------------------ I/O helpers ------------------------------------------

def load_json_map(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return {str(d["doc_id"]): d for d in json.load(f)}

def load_pair(path_a: Path, path_b: Path, originals: Optional[Path] = None) -> List[Dict[str, Any]]:
    data_a = load_json_map(path_a)
    data_b = load_json_map(path_b)
    originals_map = {}
    if originals and originals.exists():
        originals_map = {str(d["doc_id"]): d.get("source_text", "") for d in json.load(originals.open(encoding="utf-8"))}

    docs: List[Dict[str, Any]] = []
    for doc_id, rec_a in data_a.items():
        rec_b = data_b.get(doc_id, {})
        summary_a = rec_a.get("summary") or rec_a.get("error", {}).get("actual_content_summary", "")
        summary_b = rec_b.get("summary") or rec_b.get("error", {}).get("actual_content_summary", "")
        if not summary_a and not summary_b:
            # Skip records with no summaries at all
            continue
        docs.append({
            "doc_id": doc_id,
            "source_text": originals_map.get(doc_id, ""),
            "current_prompt": "",
            "summary_model_A": summary_a,
            "summary_model_B": summary_b
        })
    return docs

def save_json(obj: Dict[str, Any], out_dir: Path, doc_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"result_{doc_id}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2))

# ------------------ CLI ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Iterative Prompt Optimisation Loop – Gemini Pair Mode")
    parser.add_argument("--model_a", required=True, help="JSON file – summaries from Model A (e.g. *flash*)")
    parser.add_argument("--model_b", required=True, help="JSON file – summaries from Model B")
    parser.add_argument("--output",  required=True, help="Directory for result files")
    parser.add_argument("--original", required=False, help="Optional JSON with full source_text per doc_id")

    args = parser.parse_args()
    path_a     = Path(args.model_a).expanduser().resolve()
    path_b     = Path(args.model_b).expanduser().resolve()
    originals  = Path(args.original).expanduser().resolve() if args.original else None
    out_dir    = Path(args.output).expanduser().resolve()

    docs = load_pair(path_a, path_b, originals)
    print(f"Loaded {len(docs)} paired records.")
    for doc in tqdm(docs, desc="Processing"):
        try:
            result = run_iteration(doc)
            save_json(result, out_dir, doc['doc_id'])
        except Exception as e:
            print(f"[WARN] doc_id={doc['doc_id']} failed: {e}")

if __name__ == "__main__":
    main()
