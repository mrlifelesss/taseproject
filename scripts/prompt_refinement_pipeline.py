#!/usr/bin/env python3
"""
Prompt‑refinement pipeline – **smart about duplicate models**
===========================================================
If both weak‑model slots resolve to the **same model ID** (`X X` or legacy
`--weak_model X`), we now:

1. **Run the summariser only *once*.**  The second pass is skipped, saving
   tokens and API quota.
2. Still create two logical keys (`X__0`, `X__1`) that point to the *same*
   summary files, so the evaluator can compare “run‑A vs run‑B” if you want
   it to—or you can short‑circuit that later.

No change when you provide two distinct IDs.
"""
from __future__ import annotations
import argparse, json, math, os, re, statistics as st, subprocess, sys, time
from datetime import datetime as _dt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SUMMARIZER  = SCRIPT_DIR / "summarize_with_gemini.py"
EVALUATOR   = SCRIPT_DIR / "ipol_gemini_pair_prompts2.py"

# ───────────────────────── helper: shell runner ──────────────────────────

def _run(cmd: list[str], *, env: dict | None = None):
    print("→", " ".join(map(str, cmd)))
    cp = subprocess.run(cmd, env=env)
    if cp.returncode:
        raise RuntimeError(f"sub‑process failed ({cp.returncode}) – {' '.join(cmd)}")

# ───────────────────────── helpers: summarise & discover ─────────────────

def summarise(docs: str, prompts: str, api_key: str, model: str, out_dir: Path):
    env = os.environ.copy(); env["GEMINI_API_KEY"] = api_key
    _run([
        sys.executable, str(SUMMARIZER),
        docs, prompts,
        "--model", model,
        "--output", str(out_dir)
    ], env=env)


def discover_summary_files(model: str):
    patt = re.compile(rf"gemini_responses_{re.escape(model)}_(.+?)\.json$")
    for p in Path("gemini_outputs").glob(f"gemini_responses_{model}_*.json"):
        if m := patt.match(p.name):
            yield m.group(1), p

# ───────────────────────── helper: evaluate & update ─────────────────────

def evaluate(handle: str, path_a: Path, path_b: Path, docs: str, prompts: str,
             api_key: str, tmp_dir: Path) -> dict | None:
    env = os.environ.copy(); env["GEMINI_API_KEY"] = api_key
    out_dir = tmp_dir / f"{handle}_{int(time.time())}"; out_dir.mkdir(parents=True)

    _run([
        sys.executable, str(EVALUATOR),
        "--model_a", str(path_a),
        "--model_b", str(path_b),
        "--original", docs,
        "--prompts", prompts,
        "--output", str(out_dir)
    ], env=env)

    try:
        res_file = next(out_dir.glob("*.json"))
        return json.loads(res_file.read_text(encoding="utf-8"))
    except StopIteration:
        return None


def update_repo(repo_path: Path, handle: str, new_prompt: str) -> bool:
    data = json.loads(repo_path.read_text(encoding="utf-8"))
    arr  = data.get("prompts", data if isinstance(data, list) else [])
    changed = False
    for rec in arr:
        if rec.get("handle") == handle:
            if rec.get("prompt_text", "").strip() != new_prompt:
                rec["prompt_text"] = new_prompt; changed = True
            break
    else:
        arr.append({"handle": handle, "prompt_text": new_prompt}); changed = True
    if changed:
        repo_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return changed

# ───────────────────────── metrics helpers ───────────────────────────────

def _cohen_d(deltas):
    return 0.0 if len(deltas) < 2 else (st.mean(deltas) / st.stdev(deltas or [1]))

# ───────────────────────── main loop ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("Prompt‑refinement (duplicate‑model smart)")
    ap.add_argument("docs")  	#The JSON file that contains the full texts to summarise (e.g. classified.json).
    ap.add_argument("prompt_repo") # The prompt-repository file that holds your category prompts (e.g. prompt_depository.json). It gets updated in-place.
    ap.add_argument("--weak_models", nargs=2) #	IDs of the two weak models to generate A/B summaries (e.g. gemini-2.5-flash gemini-2.5-mini).
    ap.add_argument("--weak_model")
    ap.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY"))
    ap.add_argument("--iterations", type=int, default=1)
    args = ap.parse_args()

    # Resolve weak model list (always length‑2 for downstream logic)
    if args.weak_models:
        weak_models = list(args.weak_models)
    elif args.weak_model:
        weak_models = [args.weak_model, args.weak_model]
    else:
        ap.error("Provide --weak_models A B or --weak_model X")

    repo_path = Path(args.prompt_repo).resolve()
    work_root = Path("auto_prompt_runs") / _dt.now().strftime("%Y%m%d_%H%M%S")
    work_root.mkdir(parents=True)

    for it in range(1, args.iterations + 1):
        iter_dir = work_root / f"iter{it}"; iter_dir.mkdir()
        print(f"\n================= ITERATION {it} =================")

        # ① run summariser – but skip duplicate second call
        seen = set()
        for idx, wk in enumerate(weak_models):
            if wk in seen:
                print(f"⏩  Duplicate model '{wk}' – skipping 2nd run (idx {idx})")
                continue
            summarise(args.docs, str(repo_path), args.api_key,
                      wk, iter_dir / f"summaries_{idx}_{wk}")
            seen.add(wk)

        # ② build summary map (two logical keys even if they point to same path)
        summary_map = {}
        for idx, wk in enumerate(weak_models):
            for handle, p in discover_summary_files(wk):
                summary_map.setdefault(handle, {})[f"{wk}__{idx}"] = p

        handles_ready = [h for h, d in summary_map.items()
                         if len(d) == 2]
        if not handles_ready:
            print("⚠ No categories had summaries from both logical runs – aborting.")
            break

        eval_dir = iter_dir / "eval"; eval_dir.mkdir()
        wins, deltas, any_change = 0, [], False

        for handle in handles_ready:
            path_a, path_b = list(summary_map[handle].values())
            print(f"\n--- {handle}: evaluating {path_a.name} ↔ {path_b.name} ---")
            payload = evaluate(handle, path_a, path_b, args.docs,
                               str(repo_path), args.api_key, eval_dir)
            if not payload:
                continue

            sc_a, sc_b = payload.get("score_a"), payload.get("score_b")
            if isinstance(sc_a, (int, float)) and isinstance(sc_b, (int, float)):
                delta = sc_b - sc_a; deltas.append(delta); wins += delta > 0

            new_prompt = (payload.get("optimized_prompt") or "").strip()
            if new_prompt and update_repo(repo_path, handle, new_prompt):
                any_change = True

        if deltas:
            print(f"Win‑rate: {wins/len(deltas):.2%}  |  Mean Δ: {st.mean(deltas):+.2f}")
        if not any_change:
            print("No prompts changed – stopping early.")
            break

    print("\n🎉  Done. Updated repo at", repo_path)

if __name__ == "__main__":
    import statistics as st, math
    main()
