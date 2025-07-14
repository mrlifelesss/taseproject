import json
import argparse
from google import genai
import os
import time
from google.genai.errors import ClientError
from collections import defaultdict
import sys
from google.genai import types
import re
import functools, urllib.request, sentencepiece as spm
# --------------------------------------------------
# Script to feed classified docs into Gemini 2.5 Flash-Lite
# (No longer requires explicit project ID)
# python summarize_with_gemini.py classified.json Prompt_depository.json \
# --------------------------------------------------
SYSTEM_PROMPT = """
You are an expert data processing agent. Your operation is governed by a strict set of rules to ensure accuracy, consistency, and adherence to format.
**Rule 1: Strict JSON Output Contract** :Your final output MUST be a single, valid JSON object and nothing else. This object will contain the key `doc_id` (with the document's identifier) and EITHER a `summary` key OR an `error` key, but never both. The success case may also include additional keys like `keywords` as defined in the task instructions.
    **On Success:** The output will be `{ \"doc_id\": \"...\", \"summary\": \"...\", \"keywords\": [...] }` (if keywords are requested).**
    **On Failure:** If the document is irrelevant or a high-quality summary cannot be produced, the output MUST be `{ \"doc_id\": \"...\", \"error\": { \"type\": \"Irrelevant Document / Low Quality Summary\", \"message\": \"...\", \"actual_content_summary\": \"...\" } }`. The `actual_content_summary` key must contain a brief, one-sentence summary of the document's actual content. You will be given the specific error `message` to use in the task instructions.
**Rule 2: Sole Source of Truth**: Your output must be exclusively traceable to the provided document. Using external knowledge, making assumptions, or recalling information from past interactions is strictly forbidden. You must process the entire document, including all tables and appendices, before responding.\n\n**Rule 3: Language and Anonymity Protocol**\nAll `summary` text MUST be written in Hebrew. Crucially, you must NEVER include the name of the reporting company or the publication date of the report in the summary, ensuring the output is anonymized.
**Your Task:**:You will receive specific instructions for different summarization tasks. Follow those instructions precisely to evaluate the document, extract the required information, and construct the final JSON object according to the rules above."

"""
MAX_TOKENS_PER_MIN = 15000
QUOTA_VAL_RE = re.compile(r'"quotaValue"\s*:\s*"(\d+)"')
PER_DAY_RE   = re.compile(r"PerDay")
TOKEN_RL_RE  = re.compile(r'"retryDelay"\s*:\s*"(\d+)s"')

DEFAULT_LOCAL_PATH = (
    os.getenv("GEMMA_SPM_PATH")              # user override
    or r"G:\tase_project\models\gemma\tokenizer.model"
)

RAW_URL = (
    "https://huggingface.co/google/gemma-2b-it/"
    "resolve/main/tokenizer.model"
)  # same file for all Gemma checkpoints

def _ensure_tokenizer_file(path: str = DEFAULT_LOCAL_PATH,
                           url: str = RAW_URL) -> str:
    """
    Return *path* if the tokenizer is already on disk.
    If missing, try to download it; if that fails, raise so caller can
    fall back to a heuristic.
    """
    if os.path.exists(path):
        return path

    # create parent dir
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"⏬  downloading Gemma tokenizer model to {path} …")
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        # Delete partial file if any
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise RuntimeError(
            "Could not download tokenizer.model automatically.\n"
            "👉  Download it manually from the URL below and set the "
            "GEMMA_SPM_PATH env-var to its location.\n"
            f"   {url}\n"
            f"Original error: {e}"
        ) from e
@functools.lru_cache(maxsize=1)
def _gemma_sp():
    try:
        model_path = _ensure_tokenizer_file()
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return sp
    except Exception as e:
        # cache the *failure* to avoid retry spam
        _gemma_sp.fail_reason = e
        raise

def gemma_token_count(text: str) -> int:
    try:
        return len(_gemma_sp().encode(text))
    except Exception:
        # one warning only
        if not hasattr(_gemma_sp, "warned"):
            print(f"⚠ Gemma tokenizer unavailable ({_gemma_sp.fail_reason}); "
                  "falling back to len(text)//4 heuristic")
            _gemma_sp.warned = True
        return max(1, len(text) // 4)

class DailyQuotaExceeded(Exception):
    """Raised when daily free‐tier quota is exhausted."""
    pass
def load_prompts(prompt_depot_path: str) -> tuple[str, dict]:
    """
    Loads system prompt and task-specific prompts into a dict.
    """
    with open(prompt_depot_path, 'r', encoding='utf-8') as f:
        depot = json.load(f)
    prompts = {p['handle']: p['prompt_text'] for p in depot.get('prompts', [])}
    system_prompt = prompts.pop('system_prompt', '')
    return system_prompt, prompts
def _error_payload(e: ClientError) -> dict:
    """
    Return the error payload as a Python dict even if the SDK only gave us a
    string. Returns {} on failure.
    """
    if hasattr(e, "_to_dict"):
        try:
            return e._to_dict()
        except Exception:
            pass

    # fall-back – try to locate the first `{ … }` blob in the string
    try:
        m = re.search(r"{.*}", str(e))
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}

def safe_generate(client, system_prompt: str, model_id: str, **kwargs):
    """
    Wrapper around generate_content() that
      • skips system_instruction for Gemma models
      • sleeps retryDelay on per-minute throttle
      • raises DailyQuotaExceeded on daily cap
    """
    while True:
        try:
            cfg = {}
            if system_prompt and not model_id.startswith("gemma-"):
                cfg["system_instruction"] = system_prompt

            return client.models.generate_content(
                **kwargs,
                model=model_id,                           # in case not in kwargs
                config=types.GenerateContentConfig(**cfg) if cfg else None,
            )

        except ClientError as err:
            if err.code != 429:
                raise                                  # not a quota error

            # -------- quota handling --------
            try:
                info = err._to_dict()
            except Exception:
                info = {}

            delay = 6
            daily = False

            if info:
                delay = int(info.get("retryDelay", "6s").rstrip("s"))
                for d in info.get("details", []):
                    if d.get("@type", "").endswith("QuotaFailure"):
                        for v in d.get("violations", []):
                            if PER_DAY_RE.search(v.get("quotaId", "")):
                                daily = True
            else:
                s = str(err)
                m = TOKEN_RL_RE.search(s)
                if m:
                    delay = int(m.group(1))
                daily |= "PerDay" in s

            if daily:
                raise DailyQuotaExceeded(
                    f"Daily quota exhausted for {model_id}"
                ) from err

            delay = max(1, delay) + 1
            print(f"⚠ Per-minute quota hit ({model_id}), sleeping {delay}s …")
            time.sleep(delay)
def summarize_documents(
    docs_path: str,
    prompt_depot_path: str,
    api_key: str,
    model_id: str,
    output_path: str
):
    # ── PREPARE output_dir & checkpoint ──
    output_dir = './gemini_outputs'
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            processed_ids = set(json.load(f))
    else:
        processed_ids = set()
    print(f"Loaded {len(processed_ids)} processed doc_ids from checkpoint.")

    # Load classified docs, then filter out already‐done ones
    with open(docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    documents = [d for d in documents if d.get('doc_id')]
    total = len(documents)
    print(f"{total} docs to process after filtering previous runs.")

    # Load prompts
    system_prompt, prompt_map = load_prompts(prompt_depot_path)

    # Init GenAI client
    client = genai.Client(api_key= api_key)

    results = []
    total = len(documents)
    for idx, doc in enumerate(documents, start=1):
        doc_id = doc.get('doc_id')
        handle = doc.get('category_handle')
        text = doc.get('text', '')
        task_prompt = prompt_map.get(handle)
        if not task_prompt:
            print(f"[{idx}/{total}] Skipping doc {doc_id}: no prompt for handle '{handle}'")
            continue

        # Combine system + task prompt + document content
        full_prompt = f"{system_prompt}\n\n{task_prompt}\n\n{text}"

        # Inform about request
        print(f"[{idx}/{total}] Sending request for doc {doc_id} (handle: {handle})...")
        start_time = time.time()     
        token_estimate = len(full_prompt) // 4 or 1   # never let it be zero

        if model_id.startswith("gemma-"):
            n_tokens = gemma_token_count(full_prompt)
            if n_tokens > MAX_TOKENS_PER_MIN:
                print(f"Skipping doc {doc_id}: {n_tokens} > {MAX_TOKENS_PER_MIN}")
                continue
            
        try:
            response = safe_generate(
                client,
                system_prompt = system_prompt,
                model_id=model_id,
                contents=full_prompt
            )
        except DailyQuotaExceeded as e:
            print(f"[{idx}/{total}] {e}. Stopping early so you can resume tomorrow.")
            break
      #  except Exception as e:
       #     print(f"[{idx}/{total}] Error on doc {doc_id}: {e}")
        #    continue
        elapsed = time.time() - start_time
        print(f"[{idx}/{total}] Received response for doc {doc_id} in {elapsed:.1f}s")

        # Strip Markdown fences and parse the JSON
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].rstrip("```").strip()
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                # If it's not a dict, treat as raw response
                results.append({'doc_id': doc_id, 'raw_response': text})
                continue
        except json.JSONDecodeError:
            results.append({'doc_id': doc_id, 'raw_response': response.text})
            continue
        
        # keep only the clean fields
        entry = {'doc_id': doc_id}
        entry['category_handle'] = handle
        if 'summary' in payload:
            entry['summary'] = payload['summary']
        if 'keywords' in payload:
            entry['keywords'] = payload['keywords']
        if 'error' in payload:
            entry['error'] = payload['error']
        results.append(entry)
       # ── NEW: update checkpoint immediately on successful entry ──
        processed_ids.add(doc_id)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed_ids), f, ensure_ascii=False, indent=2)

    # Group by category_handle
    groups = defaultdict(list)
    for entry in results:
        handle = entry.get('category_handle', 'UNKNOWN')
        groups[handle].append(entry)
    # Make sure your output directory exists (or use '.' for cwd)
    os.makedirs(output_dir, exist_ok=True)

    # Write one JSON per handle
    for handle, items in groups.items():
        # sanitize handle for filenames
        safe_handle = handle.lower().replace(' ', '_')
        out_file = os.path.join(output_dir, f"gemini_responses_{model_id}_{safe_handle}.json")
        with open(out_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(items)} responses for “{handle}” → {out_file}")
    print(f"Wrote {len(results)} responses to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Summarize classified docs with Gemini 2.5 Flash-Lite'
    )
    parser.add_argument('docs', help='Path to classified JSON')
    parser.add_argument('prompts', help='Path to Prompt_depository.json')
    parser.add_argument('--api_key', default= os.getenv("GEMINI_API_KEY"), help='Google API key for GenAI')
    parser.add_argument('--model', default='gemma-3-27b-it',
                        help='Gemini model ID')
    parser.add_argument('--output', default='gemini_responses.json',
                        help='Output JSON file for summaries')
    args = parser.parse_args()
# gemini-2.5-flash gemini-2.5-flash-lite-preview-06-17
    summarize_documents(
        docs_path=args.docs,
        prompt_depot_path=args.prompts,
        api_key=args.api_key,
        model_id=args.model,
        output_path=args.output
    )
