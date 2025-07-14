#!/usr/bin/env python3
"""
process_company_disclosures.py

Scan the CompanyDisclosuresHebrew table for items missing ANY of these four attributes:
  • issuerName
  • pdf_count
  • pdf_total_size
  • events

For each such “incomplete” item:
  1. Find its HTML file in attachedFiles (the first .html/.htm).
  2. Parse that HTML with BeautifulSoup → extract fields:
       - company_name_he, company_name_en, registration_number,
         form_type, submission_date, subject, proof_id, language
  3. Count how many attachedFiles end in .pdf, and sum their sizes.
  4. Update the same DynamoDB item (key = issuerName + publicationDate) with:
       • all extracted HTML fields
       • pdf_count
       • pdf_total_size
"""

import os
import re
import logging
import concurrent.futures

from langdetect import detect, DetectorFactory, LangDetectException
import boto3
from boto3.dynamodb.conditions import Attr
from bs4 import BeautifulSoup
from decimal import Decimal

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BUCKET_NAME    = os.environ.get('S3_BUCKET', 'summery-prompts-data')
TABLE_NAME     = os.environ.get('DDB_TABLE', 'CompanyDisclosuresHebrew')
MAX_WORKERS    = 10
# ────────────────────────────────────────────────────────────────────────────────

# Initialize AWS resources
dynamodb = boto3.resource('dynamodb')
table     = dynamodb.Table(TABLE_NAME)
s3        = boto3.client('s3')

# Ensure deterministic language detection
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """
    Return 'he' only if langdetect says 'he', else return 'en'.
    If text is too short or detection fails, default to 'en'.
    """
    try:
        if len(text.strip()) < 20:
            return "en"
        lang = detect(text)
        return "he" if lang == "he" else "en"
    except LangDetectException:
        return "en"


def extract_html_fields(s3_key: str) -> dict:
    """
    Download the HTML file from S3 (key = s3_key), parse with BeautifulSoup,
    and extract these fields (if present in the page by element ID):
      - company_name_he   (element id="HeaderEntityNameEB")
      - company_name_en   (element id="HeaderEntityNameEn")
      - registration_number  (element id="HeaderSingNumberD")
      - form_type         (element id="HeaderFormNumber")
      - submission_date   (element id="HeaderSendDate")
      - subject           (element id="HeaderFormSubject")
      - proof_id          (element id="HeaderProofValue")
    Also detect overall language ('he' vs 'en') from the full text.

    Returns a dict with exactly those keys (values may be None if the ID wasn’t found).
    """
    resp = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    html = resp['Body'].read()
    soup = BeautifulSoup(html, 'html.parser')

    def get_txt(el_id: str) -> str | None:
        tag = soup.find(id=el_id)
        return tag.get_text(strip=True) if tag else None

    data = {
        'company_name_he':     get_txt('HeaderEntityNameEB'),
        'company_name_en':     get_txt('HeaderEntityNameEn'),
        'registration_number': get_txt('HeaderSingNumberD'),
        'form_type':           get_txt('HeaderFormNumber'),
        'submission_date':     get_txt('HeaderSendDate'),
        'subject':             get_txt('HeaderFormSubject'),
        'proof_id':            get_txt('HeaderProofValue'),
    }

    full_text = soup.get_text(separator=' ')
    data['language'] = detect_language(full_text)

    return data


def compute_pdf_stats(s3_keys: list[str]) -> tuple[int, int]:
    """
    From a list of S3 keys (strings), count how many end in '.pdf' (case-insensitive)
    and sum their sizes (in bytes). Returns (pdf_count, total_bytes).
    """
    pdf_keys = [k for k in s3_keys if k.lower().endswith('.pdf')]
    total_bytes = 0

    for key in pdf_keys:
        head = s3.head_object(Bucket=BUCKET_NAME, Key=key)
        total_bytes += head['ContentLength']

    return len(pdf_keys), total_bytes


def update_ddb_item(pk: str, sk: str, fields: dict, pdf_count: int, pdf_bytes: int):
    """
    Update the DynamoDB item where:
      • Partition key  = issuerName  (String)
      • Sort key       = publicationDate  (String)

    'fields' is a dict of all the HTML-extracted fields (strings or None).
    'pdf_count' and 'pdf_bytes' are ints.

    We build an UpdateExpression that sets each key. Example final expression:
      SET #company_name_he = :company_name_he,
          #company_name_en = :company_name_en,
          ...
          #pdf_count       = :pdf_count,
          #pdf_total_size  = :pdf_total_size
    """
    expr_parts = []
    expr_attr_names = {}
    expr_attr_values = {}

    # 1) Add each HTML-extracted field under its own placeholder
    for k, v in fields.items():
        # Skip None values (optional). If you want to write None explicitly, remove this check.
        if v is None:
            continue

        expr_parts.append(f"#{k} = :{k}")
        expr_attr_names[f"#{k}"] = k
        expr_attr_values[f":{k}"] = v

    # 2) Add pdf_count and pdf_total_size
    expr_parts.append("#pdf_count = :pdf_count")
    expr_attr_names["#pdf_count"] = "pdf_count"
    expr_attr_values[":pdf_count"] = Decimal(pdf_count)

    expr_parts.append("#pdf_total_size = :pdf_total_size")
    expr_attr_names["#pdf_total_size"] = "pdf_total_size"
    expr_attr_values[":pdf_total_size"] = Decimal(pdf_bytes)

    update_expr = "SET " + ", ".join(expr_parts)

    table.update_item(
        Key={
            'issuerName':      pk,
            'publicationDate': sk
        },
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_attr_names,
        ExpressionAttributeValues=expr_attr_values
    )


def process_item(item: dict):
    """
    Called on each DynamoDB item that passed the scan filter (missing at least one of the four fields).
    - Expect 'issuerName' and 'publicationDate' to be present in 'item'.
    - Expect 'attachedFiles' to be a list of S3 keys (strings or {'S': str}).
    """
    pk = item.get('issuerName')
    if pk is None:
        logging.warning(f"Skipping item (no 'issuerName'): {item}")
        return

    sk = item.get('publicationDate')
    if sk is None:
        logging.warning(f"[{pk}] skipping (no 'publicationDate'): {item}")
        return

    attached_raw = item.get('attachedFiles', [])
    attached: list[str] = []
    for entry in attached_raw:
        # DynamoDB might give you {"S": "<key>"} or directly a Python str
        if isinstance(entry, dict) and 'S' in entry:
            attached.append(entry['S'])
        elif isinstance(entry, str):
            attached.append(entry)

    # Find the first HTML/HTM key (case-insensitive)
    html_keys = [k for k in attached if re.search(r'\.html?$', k, re.IGNORECASE)]
    if not html_keys:
        logging.warning(f"[{pk}] no .htm/.html in attachedFiles → skipping")
        return

    # 1) Extract all HTML-based fields from the first HTML file
    fields = extract_html_fields(html_keys[0])

    # 2) Compute PDF stats (count + total size) across all attachedFiles
    pdf_count, pdf_bytes = compute_pdf_stats(attached)

    # 3) Update DynamoDB with all extracted fields + pdf_count/pdf_total_size
    update_ddb_item(pk, sk, fields, pdf_count, pdf_bytes)
    logging.info(f"[{pk}] updated (form_type={fields.get('form_type')}, pdfs={pdf_count})")


def main():
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(message)s"
    )

    # We want items where ANY of these four attributes is missing:
    #   issuerName   OR pdf_count   OR pdf_total_size   OR events
    #
    paginator = table.meta.client.get_paginator('scan')
    scan_kwargs = {
        'TableName': TABLE_NAME,
        'FilterExpression': (
            Attr('form_type').not_exists()
            | Attr('pdf_count').not_exists()
            | Attr('pdf_total_size').not_exists()
            | Attr('events').not_exists()
        ),
        # We need these fields in 'item' so that process_item() can run:
        'ProjectionExpression': 'issuerName, publicationDate, attachedFiles'
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []

        for page in paginator.paginate(**scan_kwargs):
            for item in page.get('Items', []):
                # Submit each “incomplete” item to a worker thread
                futures.append(pool.submit(process_item, item))

        # Wait for all threads to finish (and propagate any exceptions)
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception:
                logging.exception("Error in worker thread")


if __name__ == "__main__":
    main()
