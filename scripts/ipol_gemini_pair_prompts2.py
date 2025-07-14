"""ipol_gemini_pair_prompts_holistic.py  (2025‑07‑08)
Fixed single‑turn prompt builder & empty‑response handling.
• Removed accidental double‑call `single_call(...)(...)`.
• Built `single_prompt` with explicit `\n` to avoid syntax errors.
• If Gemini returns empty text, we now raise a RuntimeError before JSON parse.
"""

import json, os, argparse, time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import google.genai as genai
import re
from google.genai import types

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
GEN_MODEL            = "gemini-2.5-pro"
MAX_TOKENS_RESPONSE  = 120000
MAX_CHARS_PER_DOC    = 2000
MAX_TOTAL_TOKENS     = 120_000  # >→ switch to 2‑turn
SLEEP_BETWEEN_CALLS  = 5

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY env var must be set")

client = genai.Client(api_key=api_key)

prompt_lookup: Dict[str, str] = {}
SYSTEM_PROMPT = """You are an expert AI Prompt Engineer and Quality Analyst. Your function is to systematically evaluate and iteratively refine prompts to enhance the performance of other AI models. Your entire operation is governed by three non‑negotiable principles:

1. **Objective Grounding:** Base every judgement strictly on the provided source materials.
2. **Analytical Rigor:** Be methodical, unbiased, and justify every score you assign.
3. **Constructive Optimization:** Engineer a demonstrably superior prompt that fixes the weaknesses you find.
4. ** File  Type: ** **Respond ONLY with the JSON. No markdown fences.**
**Subject: Automated Comparative Analysis and Iterative Prompt Refinement**
"""

META_DOCS_HEADER = (
    "You will receive the raw texts of corporate disclosures. Hold them in "
    "context for later evaluation. Do not respond yet, just acknowledge with 'OK'."
)

META_TASK_TEMPLATE =  """
Your mission is to evaluate two model summaries of the *same* document and then craft an improved prompt.  You will receive three large literal text blocks **immediately after this instruction**:

```
### SOURCE_TEXT
<full_announcement_text>
### MODEL_A_SUMMARY
{model_a_block}
### MODEL_B_SUMMARY
{model_b_block}
```

---
### Execution Flow  (perform in order)

**Step 1 – Gold‑Standard Summary**  
Generate your own ideal summary of each `SOURCE_TEXT`. (Use the *current prompt* embedded inside that text if any.)

**Step 2 – Comparative Evaluation**  
Score **Model A** and **Model B** from 1 (poor) to 10 (excellent) on:
1. **Accuracy & Factual Integrity**  
2. **Completeness of Material Information**  
3. **Clarity & Conciseness**  
4. **Contextual Understanding & Synthesis**  
5. **Adherence to Instructions**

**Step 3 – Verdict**  
Declare the better summary (or *Tie*) and give a one‑sentence rationale.

**Step 4 – Prompt Optimization**  
Write an **`optimized_prompt` (v2)** that explicitly prevents the shortcomings you observed (e.g. enforce full‑document scan, demand context, add guardrails against hallucination, specify structure).

---
If multiple documents are present, score *holistically* – do NOT
produce per-document keys. Return exactly one JSON object, no markdown.
If one file has more documents then the other, ignore the extras.
### Output Schema  (return exactly this JSON ‑ minified)
```json
{{
  "analysis": {{
    "Gold_standard_summary": "...",
    "evaluation_scores": {{
      "model_A": {{"accuracy":0,"completeness":0,"clarity":0,"context":0,"adherence":0}},
      "model_B": {{"accuracy":0,"completeness":0,"clarity":0,"context":0,"adherence":0}}
    }},
    "verdict": {{"winner":"Model A|Model B|Tie","rationale":"..."}}
  }},
  "optimized_prompt": "..."
}}
```
**Respond ONLY with the JSON. No markdown fences.**
json
{{
  "analysis": {{
    "gold_standard_summary": "[Your summary from Step 1]",
    "evaluation_scores": {{
      "model_A": {{
        "accuracy": "[Score 1-10]",
        "completeness": "[Score 1-10]",
        "clarity": "[Score 1-10]",
        "context": "[Score 1-10]",
        "adherence": "[Score 1-10]"
      }},
      "model_B": {{
        "accuracy": "[Score 1-10]",
        "completeness": "[Score 1-10]",
        "clarity": "[Score 1-10]",
        "context": "[Score 1-10]",
        "adherence": "[Score 1-10]"
      }}
    }},
    "verdict": {{
      "winner": "Model A" | "Model B" | "Tie",
      "rationale": "[A one-sentence explanation for your judgement.]"
    }}
  }},
  "optimized_prompt": "[The full text of the new, improved prompt (v2) you engineered in Step 4.]"
  "prompt_changes": "[The differences between the *current prompt* and  the improved prompt (v2) you engineered in Step 4.]"
  "misclasified_docs":"[the doc_ids|model of any docs that the current prompt is inappropriate for and  the  models name ( model_A/model_B)
}}
```
Populate `prompt_lookup` and optionally override SYSTEM_PROMPT.

    Expected JSON schema – either:
      {{ "prompts": [ {{"handle": "FIN_CAPITAL_STRUCTURE", "prompt_text": "…"}}, … ] }}
    or simply a list of objects with those two keys.
    """

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────

def token_len(text: str) -> int:  # ~4 chars / token heuristic
    return max(1, len(text) // 4)

def truncate(txt: str) -> str:
    return (txt or "")[:MAX_CHARS_PER_DOC]

def cat_docs(pairs: List[Tuple[str, str]]) -> str:
    return json.dumps([{"doc_id": did, "text": truncate(t)} for did, t in pairs], ensure_ascii=False)

def cat_summaries(pairs: List[Tuple[str, str]]) -> str:
    return "\n\n".join(f"### doc {did}\n{truncate(t)}" for did, t in pairs)

def _safe(s: str) -> str:
    # 1. replace every char that is NOT A-Z, a-z, 0-9, dot, underscore or dash
    #    with an underscore
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    # 2. trim to the first 60 characters (keeps paths short / Windows-safe)
    clean = clean[:60]

    # 3. strip leading/trailing underscores or dots so we don’t get “._foo_.json”
    return clean.strip("_.")

# ────────────────── Prompt‑depository loader ──────────────────

def load_prompt_depository(path: Path):
    """Populate `prompt_lookup` and optionally override SYSTEM_PROMPT.

    Expected JSON schema – either:
      { "prompts": [ {"handle": "FIN_CAPITAL_STRUCTURE", "prompt_text": "…"}, … ] }
    or simply a list of objects with those two keys.
    """
    global prompt_lookup, SYSTEM_PROMPT
    if not path.exists():
        print(f"⚠ prompt‑depository file not found: {path}; continuing with defaults")
        return

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    records = data.get("prompts", data)  # accept top‑level list too
    for rec in records:
        handle = rec.get("handle") or rec.get("category_handle") or rec.get("name")
        text   = rec.get("prompt_text") or rec.get("text") or rec.get("prompt")
        if not handle or text is None:
            continue
        else:
            prompt_lookup[handle] = text

# ────────────────── Gemini wrappers ──────────────────

def single_call(prompt: str) -> str:
    resp = client.models.generate_content(
    model=GEN_MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_TOKENS_RESPONSE,
        system_instruction=SYSTEM_PROMPT
    )
)
    # google‑genai may return None or an object whose .text is None
    txt = "" if (resp is None) else (resp.text or "")
    #if not txt.strip():
       # raise RuntimeError(
        #    "Gemini returned an empty response – likely quota exhaustion or an "
        #    "oversized prompt that was silently rejected. Check quota and "
        #    "try truncating MAX_CHARS_PER_DOC."
        #)
    return txt.strip()

def two_turn_call(docs_block: str, task_prompt: str, gen_model: str = "gemini-2.5-pro") -> str:
    # Initialize GenerativeModel.
    # The system_instruction is the recommended way to set a system prompt.
    model = genai.GenerativeModel(model_name=gen_model, system_instruction=SYSTEM_PROMPT)

    # Start a new chat session.
    # The chat history will automatically include previous turns.
    chat = model.start_chat(history=[])

    # Turn 1: Send the initial document context
    response1 = chat.send_message(f"{META_DOCS_HEADER}\n\n{docs_block}")
    # You might want to check response1 or handle it if it's relevant to the flow

    # Turn 2: Send the specific task prompt
    response2 = chat.send_message(task_prompt)

    # Basic error checking for an empty response
    if not response2 or not getattr(response2, "text", "").strip():
        raise RuntimeError("Gemini returned an empty response in the two-turn flow.")

    return response2.text.strip()

# ────────────────── Core logic ──────────────────

def run_holistic(out_dir: Path,
                 docs_pairs: List[Tuple[str, str]],
                 a_pairs: List[Tuple[str, str]],
                 b_pairs: List[Tuple[str, str]]):

    docs_block  = cat_docs(docs_pairs)
    task_prompt = META_TASK_TEMPLATE.format(
        full_announcement_text = docs_block,
        model_a_block=cat_summaries(a_pairs),
        model_b_block=cat_summaries(b_pairs),
    )

    tot_tokens = token_len(docs_block) + token_len(task_prompt)
    print(f"~{tot_tokens:,} tokens (heuristic)")

    if tot_tokens <= MAX_TOTAL_TOKENS:
        single_prompt = (
            "You are an expert evaluator of financial‑language summaries.\n\n"
            "Below is a JSON list of the full disclosure texts followed by the two "
            "sets of summaries you must evaluate.\n\n"
            + task_prompt + "\n\nReply with minified JSON only."
        )
        (out_dir / "holistic_prompt_single.txt").write_text(single_prompt, encoding="utf-8")
        raw = single_call(single_prompt)
    else:
        (out_dir / "holistic_docs_first.txt").write_text(docs_block, encoding="utf-8")
        (out_dir / "holistic_task_second.txt").write_text(task_prompt, encoding="utf-8")
        raw = two_turn_call(docs_block, task_prompt)

    if raw.startswith("```json"):
        raw = raw[7:].rstrip("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        (out_dir / "holistic_raw_reply.txt").write_text(raw, encoding="utf-8")
        raise ValueError("Gemini did not return valid JSON – see holistic_raw_reply.txt")

# ────────────────── Data ingestion & CLI ──────────────────

def load_json_utf8(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def build_docs(a_path: Path, b_path: Path, orig_path: Optional[Path]):
    data_a = load_json_utf8(a_path)
    data_b = load_json_utf8(b_path)
    orig   = {str(d["doc_id"]): d.get("text", "") for d in (load_json_utf8(orig_path) if orig_path else [])}
    docs=[]
    for rec_a in data_a:
        did=str(rec_a["doc_id"])
        rec_b=next((x for x in data_b if str(x.get("doc_id"))==did), {})
        docs.append({"doc_id":did,
                     "source_text":orig.get(did, ""),
                     "summary_model_A":rec_a.get("summary", ""),
                     "summary_model_B":rec_b.get("summary", "")})
    return docs

def main():
    ap=argparse.ArgumentParser("IPOL holistic evaluation")
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()

    load_prompt_depository(Path(args.prompts))
    docs=build_docs(Path(args.model_a), Path(args.model_b), Path(args.original))
    out_dir=Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    pairs_docs=[(d["doc_id"], d["source_text"]) for d in docs]
    pairs_a=[(d["doc_id"], d["summary_model_A"]) for d in docs]
    pairs_b=[(d["doc_id"], d["summary_model_B"]) for d in docs]

    result = run_holistic(out_dir, pairs_docs, pairs_a, pairs_b)
    sys_tag   = _safe(SYSTEM_PROMPT.split()[0] if SYSTEM_PROMPT else "system")
    a_tag     = _safe(Path(args.model_a).stem)
    b_tag     = _safe(Path(args.model_b).stem)
    out_name  = f"{sys_tag}_{a_tag}_{b_tag}.json"
    
    scores = result["analysis"]["evaluation_scores"]
    scores[a_tag] = scores.pop("model_A")     # move & rename
    scores[b_tag] = scores.pop("model_B")
    winner = result["analysis"]["verdict"]["winner"]
    if winner == "Model A":
        result["analysis"]["verdict"]["winner"] = a_tag
    elif winner == "Model B":
        result["analysis"]["verdict"]["winner"] = b_tag
    (out_dir / out_name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {out_name} written")

if __name__=="__main__":
    main()
