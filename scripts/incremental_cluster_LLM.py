import os
import json
import argparse
import pandas as pd
import google.generativeai as genai
import time
import re
import random
import json
from collections import defaultdict

def strip_wrappers(text: str) -> str:
    """
    Remove code‐fences and stray 'json' labels from the model reply
    """
    t = text.strip()
    t = re.sub(r'^```json\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^```',      '', t)
    t = re.sub(r'^json\s*',  '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*```$',   '', t)
    return t.strip()
    
def classify_or_new(doc_excerpt: str, clusters: list, model_name: str) -> tuple[int | None, str]:
    """
    Use an LLM to classify the excerpt into existing clusters or create a new one.
    Returns (cluster_id, new_title) where cluster_id is None for a new cluster.
    """
    prompt = f"I have an incoming announcement excerpt:\n\n\"{doc_excerpt}\"\n\n"
    prompt += "Existing clusters:\n"
    for c in clusters:
        prompt += f"{c['id']}. {c['title']}\n"
        for ex in c['examples']:
            prompt += f" • \"{ex}\"\n"
        prompt += "\n"
    prompt += (
    "ענה רק בעברית.\n"
    "\n"
    "להלן הכללים המדויקים – עליך לבחור רק באחת מהאפשרויות:\n"
    "1) אם ההודעה שייכת לאחד הקלסטרים הקיימים, השב בדיוק בפורמט הבא (בלי שום טקסט נוסף):\n"
    "   {\"cluster_id\": <מספר קלסטר>, \"confidence\": <1-5>}\n"
    "   **אבל** השתייכות לקלסטר קיים תתבצע רק אם confidence הוא 4 או 5.\n"
    "\n"
    "2) אם ההודעה אינה מתאימה לאף קלסטר קיים, השב בדיוק בפורמט הבא (בלי שום טקסט נוסף):\n"
    "   {\"cluster_id\": \"NEW_CLUSTER\", \"title\": \"<כותרת קצרה בעברית>\"}\n"
    "\n"
    "לא תוסיף, תוריד או תשנה דבר מלבד אובייקט JSON חוקי אחד. התשובה חייבת להיות **רק** ה-JSON, ללא כל דבר נוסף."
)

    raw = genai.GenerativeModel(model_name) \
               .generate_content(prompt).text
    reply = strip_wrappers(raw)

    # 1) Try strict JSON parse
    try:
        info = json.loads(reply)
        cid = info.get("cluster_id")
        # normalize types
        if cid == "NEW_CLUSTER":
            return None, info.get("title", "").strip()
        cid = int(cid)
        # enforce confidence ≥4
        conf = int(info.get("confidence", 0))
        if conf < 4:
            return None, ""
        return cid, ""
    except Exception:
        pass

    # 2) Look for explicit JSON‐style fields inside text
    #    e.g.  "cluster_id": 3,  "confidence": 5
    m_cid = re.search(r'cluster_id"\s*:\s*"(NEW_CLUSTER|\d+)"', reply)
    if not m_cid:
        m_cid = re.search(r'cluster_id"\s*:\s*(NEW_CLUSTER|\d+)', reply)
    if m_cid:
        val = m_cid.group(1)
        if val == "NEW_CLUSTER":
            # try to find title field
            m_title = re.search(r'title"\s*:\s*"(.*?)"', reply)
            return None, (m_title.group(1) if m_title else "")
        else:
            cid = int(val)
            # optional confidence
            m_conf = re.search(r'confidence"\s*:\s*(\d)', reply)
            conf  = int(m_conf.group(1)) if m_conf else 5
            if conf < 4:
                return None, ""
            return cid, ""

    # 3) Look for leading numeric assignment: "3", "3." or "3: Title…"
    m = re.match(r'^(\d+)(?:[.:]\s*(.+))?', reply)
    if m:
        cid = int(m.group(1))
        return cid, ""

    # 4) Look for NEW_CLUSTER:\s*"Title…"
    m = re.match(r'NEW_CLUSTER\s*[:\-]\s*"(.*)"', reply, flags=re.IGNORECASE)
    if m:
        return None, m.group(1).strip()

    # 5) No parse → treat as new cluster (best effort)
    return None, ""

def main():
    parser = argparse.ArgumentParser(description="Incremental clustering via LLM on specific doc_ids")
    parser.add_argument("--clean-dir", required=True, help="Folder of cleaned txt files")
    parser.add_argument("--csv-path", required=True, help="CSV file listing doc_id to process")
    parser.add_argument("--model-name", default="gemini-2.0-flash-lite", help="Gemini model name")
    parser.add_argument("--api-key",default  = "AIzaSyA1NptQkNBxjgFG9Lop_NrfjVKq7-qg5HU", help="LLM API key")
    parser.add_argument("--excerpt-length", type=int, default=500, help="Max chars per excerpt")
    parser.add_argument("--output", default="incremental_clusters.json", help="Output JSON file")
    args = parser.parse_args()

    # Configure Gemini
    genai.configure(api_key=args.api_key)

    # Read doc_ids from CSV
    df = pd.read_csv(args.csv_path)
    if 'doc_id' not in df.columns:
        raise ValueError(f"'doc_id' column not found in {args.csv_path}")
    doc_ids = df['doc_id'].dropna().astype(int).unique().tolist()
    random.shuffle(doc_ids)       # mixes order

    clusters = []

    for doc_id in doc_ids:
        fname = f"{doc_id}.txt"
        path = os.path.join(args.clean_dir, fname)
        if not os.path.exists(path):
            print(f"⚠️ File not found for doc_id {doc_id}, skipping")
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(args.excerpt_length)
        excerpt = text.strip().replace("\n", " ")

        cid, new_title = classify_or_new(excerpt, clusters, args.model_name)
        time.sleep(8)
        if cid is None:
            new_id = max((c['id'] for c in clusters), default=-1) + 1
            clusters.append({
                "id": new_id,
                "title": new_title or f"Cluster {new_id}",
                "examples": [doc_id]
            })
            print(f"Created new cluster {new_id}: {new_title}")
        else:
            for c in clusters:
                if c['id'] == cid:
                    c['examples'].append(excerpt)
                    print(f"Added doc_id {doc_id} to cluster {cid}: {c['title']}")
                    break

    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(clusters, out, ensure_ascii=False, indent=2)
    print(f"Clusters written to {args.output}")

if __name__ == "__main__":
    main()

