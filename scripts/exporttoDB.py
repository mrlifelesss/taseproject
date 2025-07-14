import os
import json
import requests
import urllib.parse
import mimetypes
import boto3
import  .futures
from botocore.client import Config
from botocore.exceptions import ClientError

# ─── CONFIG ────────────────────────────────────────────────────────────────
S3_BUCKET          = "summery-prompts-data"
JSON_PREFIX        = "apicallsdata/"
ATTACHMENTS_PREFIX = "apicallsdata/attachedfiles"
DDB_TABLE_NAME     = "CompanyDisclosures"
AWS_REGION         = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ─── AWS CLIENTS ────────────────────────────────────────────────────────────
session = boto3.Session(region_name=AWS_REGION)
s3      = session.client("s3")
ddb     = session.resource("dynamodb")
table   = ddb.Table(DDB_TABLE_NAME)
paginator = s3.get_paginator("list_objects_v2")
THRESHOLD = 1621489  
# ─── PROCESS JSON FILES FROM S3 ─────────────────────────────────────────────
def process_record(rec):
    report_id = rec.get("mayaReportId")
    pub_date  = rec.get("publicationDate")
    if report_id is None or pub_date is None:
        return

    # skip anything ≤ threshold
    try:
        if int(report_id) <= THRESHOLD:
            print(f"⏭ skipping reportId={report_id} (≤ {THRESHOLD})")
            return
    except ValueError:
        print(f"⚠ invalid reportId={report_id}, skipping")
        return

    report_id = str(report_id)  # ensure string for DynamoDB
    s3_keys = []

    # ─── DOWNLOAD & UPLOAD ALL ATTACHMENTS ────────────────────────────────
    for attach in rec.get("attachedfiles", []):
        url = (attach.get("url") or "").strip()
        if not url:
            continue

        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"   ✖ download failed [{url}]: {e}")
            continue

        filename = os.path.basename(urllib.parse.urlparse(url).path)
        dest_key = f"{ATTACHMENTS_PREFIX}/{report_id}/{filename}"

        # pick a content-type
        content_type = (
            r.headers.get("Content-Type")
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )

        try:
            s3.put_object(
                Bucket      = S3_BUCKET,
                Key         = dest_key,
                Body        = r.content,
                ContentType = content_type
            )
            s3_keys.append(dest_key)
            print(f"   ✔ uploaded {filename} → s3://{S3_BUCKET}/{dest_key}")
        except ClientError as e:
            print(f"   ✖ s3 upload failed for {dest_key}: {e}")

    # ─── WRITE ITEM TO DYNAMODB ──────────────────────────────────────────
    item = {
        "reportId":        report_id,
        "publicationDate": pub_date,
        "title":           rec.get("title"),
        "url":             rec.get("url"),
        "isPriority":      rec.get("isPriorityReport", False),
        "isCorrection":    rec.get("isCorrection", False),
        "issuers":         [i.get("issuerName") for i in rec.get("issuer", [])],
        "events":          [e.get("eventName")   for e in rec.get("events", [])],
    }
    if s3_keys:
        item["attachedFiles"] = s3_keys

    try:
        table.put_item(Item=item)
        print(f"→ Wrote report {report_id}@{pub_date} with {len(s3_keys)} attachments")
    except ClientError as e:
        print(f"   ✖ DynamoDB put_item failed for {report_id}: {e}")

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        # list all JSON files under your prefix
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=JSON_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".json"):
                    continue

                print(f"→ Loading JSON: {key}")
                try:
                    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                    data = json.loads(resp["Body"].read().decode("utf-8"))
                except ClientError as e:
                    print(f"   ✖ failed to fetch {key}: {e}")
                    continue

                records = data.get("mayaReports", {}).get("result", [])
                for rec in records:
                    # schedule each record for processing
                    futures.append(executor.submit(process_record, rec))

        # wait for all to finish (and surfacing errors)
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"⚠ Thread error: {e}")

if __name__ == "__main__":
    main()