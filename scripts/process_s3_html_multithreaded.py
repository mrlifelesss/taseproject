import boto3
import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup
bucket = "summery-prompts-data"
prefix = "apicallsdata/attachedfiles/"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

s3 = boto3.client("s3")

# HTML extractor
def extract_html_fields(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    selectors = {
        "company_name_he": "#HeaderEntityNameEB",
        "company_name_en": "#HeaderEntityNameEn",
        "registration_number": "#HeaderSingNumberD",
        "form_type": "#HeaderFormNumber",
        "submission_date": "#HeaderSendDate",
        "subject": "#HeaderFormSubject",
        "proof_id": "#HeaderProofValue"
    }

    header_info = {}
    for key, selector in selectors.items():
        el = soup.select_one(selector)
        if el:
            header_info[key] = el.get_text(strip=True)
            el.decompose()
        else:
            header_info[key] = None

    body_texts = []
    for tag in soup.find_all(["textarea", "span"]):
        text = tag.get_text(strip=True)
        if text and not text.startswith("__") and len(text) > 2:
            body_texts.append(text)

    clean_text = "\n".join(body_texts)
    return header_info, clean_text

# Process one S3 file
def process_s3_key(key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("windows-1255", errors="ignore")
        header, clean_body = extract_html_fields(content)

        # Use S3 key as filename base
        filename = key.replace(prefix, "").replace("/", "_").rsplit(".", 1)[0]

        json_path = output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(header, f, ensure_ascii=False, indent=2)

        txt_path = output_dir / f"{filename}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_body)

        print(f"✅ {key} → {filename}.json + .txt")
    except Exception as e:
        print(f"❌ Error processing {key}: {e}")

# Main function with multithreading
def run_multithreaded_extraction():
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys_to_process = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".html", ".htm")):
                keys_to_process.append(key)

    print(f"🔍 Found {len(keys_to_process)} HTML files.")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_s3_key, key) for key in keys_to_process]
        for _ in as_completed(futures):
            pass  # All print happens inside `process_s3_key`

if __name__ == "__main__":
    run_multithreaded_extraction()
