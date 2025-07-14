"""ipol_gemini_pair_prompts.py
Gemini IPOL – **with external prompt depository**

Usage is identical to ipol_gemini_pair.py, plus:

    --prompts Prompt_depository.json

The script injects `current_prompt` for each document according to its
`category_handle` (if present) by looking it up in the depository.  If a
handle is missing it falls back to the generic system‑level prompt.

python ipol_gemini_pair_prompts.py \
    --model_a  gemini_responses_candidate_credit_rating_flash.json \
    --model_b  gemini_responses_candidate_credit_rating.json \
    --prompts  Prompt_depository.json \
    --output   out/

"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from tqdm import tqdm
import google.genai as genai
# -------------------------------------------------------------------------
gen_model = "gemini-2.5-pro"
DEFAULT_TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gemini-2.5-pro")

# Fixed API key handling
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable must be set")

DEFAULT_TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gemini-2.5-pro")

# Initialize client with proper API key
client = genai.Client(api_key=api_key)

# These will be filled after reading the depository -----------------------
prompt_lookup: Dict[str, str] = {}

SYSTEM_PROMPT = """You are an expert AI Prompt Engineer and Quality Analyst. Your function is to systematically evaluate and iteratively refine prompts to enhance the performance of other AI models. Your entire operation is governed by three non-negotiable principles:

1.  **Objective Grounding:** Your analysis and judgment must be based exclusively and verifiably on the source materials provided in each task (e.g., the `source_text`). You are a scientific evaluator, not a subjective critic.

2.  **Analytical Rigor:** You must be methodical, unbiased, and justify your conclusions. Your evaluations must follow the specific criteria and steps outlined in the mission briefing you receive.

3.  **Constructive Optimization:** Your ultimate goal is not merely to identify failures but to engineer a demonstrably superior prompt that directly mitigates those weaknesses using established prompt engineering techniques.

"""

META_PROMPT_TEMPLATE = """
This prompt defines the specific, complex task of judging other models and optimizing a prompt. It assumes the AI is already operating under the System Prompt above.

**Subject: Automated Comparative Analysis and Iterative Prompt Refinement**

Your mission is to act as the core of an automated prompt optimization loop. You will be provided with a source document, a prompt used to summarize it, and the results from two external models. Your task is to evaluate the results, declare a winner, and engineer a superior prompt.

**Input Schema:**

You will receive a JSON object containing:
*   `doc_id`: The document's unique identifier.
*   `source_text`: The full, original text of the announcement.
*   `current_prompt`: The prompt (`v1`) given to the external models.
*   `summary_model_A`: The output from Model A.
*   `summary_model_B`: The output from Model B.

**Execution Flow (Perform these 4 steps in sequence):**

**Step 1: Generate "Gold Standard" Summary**
Using the `current_prompt` and `source_text`, generate your own ideal summary. This will serve as the baseline for your evaluation.

**Step 2: Comparative Evaluation**
Evaluate all three summaries (your Gold Standard, Model A's, and Model B's) across the following five categories, providing a score from 1 (terrible) to 10 (perfect) for **Model A** and **Model B**:

1.  **Accuracy & Factual Integrity:** Presence of any factual errors or hallucinations. (A single hallucination results in a score of 1).
2.  **Completeness of Material Information:** Inclusion of all critical business information from the entire document.
3.  **Clarity & Conciseness:** Readability and efficiency of the language.
4.  **Contextual Understanding & Synthesis:** Demonstration of understanding the "why" behind the facts, not just listing them.
5.  **Adherence to Instructions:** Success in following all rules within the `current_prompt`.

**Step 3: Judgement and Rationale**
Declare which summary (`summary_model_A` or `summary_model_B`) is superior. Provide a brief, one-sentence rationale.

**Step 4: Prompt Optimization**
Based on the identified weaknesses in the external models' summaries, engineer an `optimized_prompt` (`v2`). The new prompt must be specifically designed to prevent the observed failures. Use expert techniques such as:
*   **Forcing a complete scan** to prevent "lazy" summaries.
*   **Demanding contextual enrichment** to prevent superficial outputs.
*   **Strengthening guardrails** to prevent hallucinations.
*   **Adding dynamic formatting rules and comparative examples** to improve structure and quality.

**Output Schema:**

Your final output must be a single JSON object with the following structure:
```json
{
  "doc_id": "[doc_id from input]",
  "analysis": {
    "gold_standard_summary": "[Your summary from Step 1]",
    "evaluation_scores": {
      "model_A": {
        "accuracy": "[Score 1-10]",
        "completeness": "[Score 1-10]",
        "clarity": "[Score 1-10]",
        "context": "[Score 1-10]",
        "adherence": "[Score 1-10]"
      },
      "model_B": {
        "accuracy": "[Score 1-10]",
        "completeness": "[Score 1-10]",
        "clarity": "[Score 1-10]",
        "context": "[Score 1-10]",
        "adherence": "[Score 1-10]"
      }
    },
    "verdict": {
      "winner": "Model A" | "Model B" | "Tie",
      "rationale": "[A one-sentence explanation for your judgement.]"
    }
  },
  "optimized_prompt": "[The full text of the new, improved prompt (v2) you engineered in Step 4.]"
}
```

"""

system_prompt_text: str = SYSTEM_PROMPT

# ------------------ Helper functions -------------------------------------

def load_prompt_depository(path: Path):
    global prompt_lookup, system_prompt_text
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for rec in data["prompts"]:
        handle = rec["handle"]
        text = rec["prompt_text"]
        if handle == "system_prompt":
            system_prompt_text = text
        else:
            prompt_lookup[handle] = text

def lookup_prompt(handle: str) -> str:
    return prompt_lookup.get(handle, "")

def call_gemini_chat(model_name, messages, temperature=0.0, max_tokens=4096):
    """
    Fixed function to properly call Gemini API
    """
    try:
        # Create the full prompt from messages
        full_prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                full_prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "model":
                full_prompt += f"Assistant: {msg['content']}\n\n"
        full_prompt += "“Reply with a single, minified JSON object. Do not add ```json fences, Markdown, bullets or explanatory text. Use double quotes for every key and string value.”"
        # Make the API call using the correct method
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        
        # Extract the text from response
        return response.text.strip()
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise

def run_iteration(doc: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "user",  "content": META_PROMPT_TEMPLATE},
        {"role": "model", "content": "Awaiting JSON."},
        {"role": "user",  "content": json.dumps(doc, ensure_ascii=False)}
    ]
    
    try:
        raw = call_gemini_chat(
            DEFAULT_TEACHER_MODEL,
            messages,
        )
        
        # Strip markdown fences if present
        if raw.startswith("```json"):
            raw = raw[7:].rstrip("```").strip()
        
        return json.loads(raw)
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {raw}")
        raise
    except Exception as e:
        print(f"Error in run_iteration: {e}")
        raise

# ------------------ IO helpers (re‑used) ---------------------------------

def load_json_map(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return {str(d["doc_id"]): d for d in json.load(f)}

def build_docs(path_a: Path, path_b: Path, originals: Optional[Path]) -> List[Dict[str, Any]]:
    data_a = load_json_map(path_a)
    data_b = load_json_map(path_b)
    originals_map = {}
    if originals and originals.exists():
        originals_map = {str(d["doc_id"]): d.get("text", "") for d in json.load(originals.open(encoding="utf-8"))}

    docs = []
    for doc_id, rec_a in data_a.items():
        rec_b = data_b.get(doc_id, {})
        summary_a = rec_a.get("summary") or rec_a.get("error", {}).get("actual_content_summary", "")
        summary_b = rec_b.get("summary") or rec_b.get("error", {}).get("actual_content_summary", "")
        if not summary_a and not summary_b:
            continue
        handle = rec_a.get("category_handle") or rec_b.get("category_handle") or ""
        docs.append({
            "doc_id": doc_id,
            "source_text": originals_map.get(doc_id, ""),
            "current_prompt": lookup_prompt(handle),
            "summary_model_A": summary_a,
            "summary_model_B": summary_b
        })
    return docs

def save_json(obj: Dict[str, Any], out_dir: Path, doc_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"result_{doc_id}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2))

# ------------------ CLI ---------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="IPOL – Gemini pair mode with prompt lookup")
    p.add_argument("--model_a", required=True)
    p.add_argument("--model_b", required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--original", required=False)
    p.add_argument("--prompts", required=True, help="Prompt depository JSON")
    args = p.parse_args()

    # Check if API key is set
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return

    load_prompt_depository(Path(args.prompts).expanduser().resolve())

    docs = build_docs(Path(args.model_a), Path(args.model_b),
                      Path(args.original).expanduser().resolve() if args.original else None)

    out_dir = Path(args.output).expanduser().resolve()
    print(f"Loaded {len(docs)} docs, starting IPOL…")
    for d in tqdm(docs):
        try:
            res = run_iteration(d)
            save_json(res, out_dir, d["doc_id"])
        except Exception as e:
            print(f"[WARN] {d['doc_id']} failed: {e}")
        time.sleep(15)

if __name__ == "__main__":
    main()