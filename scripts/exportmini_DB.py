import os
import json
import requests
import urllib.parse
import mimetypes
import boto3
from botocore.exceptions import ClientError

# ─── CONFIG ────────────────────────────────────────────────────────────────
S3_BUCKET          = "summery-prompts-data"
JSON_PREFIX        = "apicallsdata/"
ATTACHMENTS_PREFIX = "apicallsdata/attachedfiles"
DDB_TABLE_NAME     = "CompanyDisclosuresHebrew"  # ← table with PK=issuerName, SK=publicationDate
AWS_REGION         = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

THRESHOLD          = 0     # (optional) skip any reportId ≤ this
MAX_ITEMS          = 2000        # stop after inserting 2,000 items total

# ─── AWS CLIENTS ────────────────────────────────────────────────────────────
session   = boto3.Session(region_name=AWS_REGION)
s3        = session.client("s3")
ddb       = session.resource("dynamodb")
table     = ddb.Table(DDB_TABLE_NAME)
paginator = s3.get_paginator("list_objects_v2")


def main():
    inserted_count = 0

    # Paginate through all JSON files under the prefix
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=JSON_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".json"):
                continue

            # If we've reached MAX_ITEMS, exit entirely
            if inserted_count >= MAX_ITEMS:
                print(f"✔ Reached {MAX_ITEMS} items. Stopping.")
                return

            print(f"→ Loading JSON: {key}")
            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                data = json.loads(resp["Body"].read().decode("utf-8"))
            except ClientError as e:
                print(f"   ✖ failed to fetch {key}: {e}")
                continue

            records = data.get("mayaReports", {}).get("result", [])
            for rec in records:
                pub_date  = rec.get("publicationDate")
                report_id = rec.get("mayaReportId")

                if pub_date is None:
                    continue

                # Skip if reportId ≤ THRESHOLD (optional)
                try:
                    if report_id is not None and int(report_id) <= THRESHOLD:
                        print(f"⏭ skipping reportId={report_id} (≤ {THRESHOLD})")
                        continue
                except ValueError:
                    print(f"⚠ invalid reportId={report_id}, skipping")
                    continue

                # 1) Download & re-upload attachments
                s3_keys = []
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

                # 2) Build events map once for this record
                events_map = {}
                for e in rec.get("events", []):
                    name = e.get("eventName")
                    if name:
                        events_map[name] = e

                # 3) Write one DynamoDB item per issuer
                for issuer_obj in rec.get("issuer", []):
                    if inserted_count >= MAX_ITEMS:
                        print(f"✔ Reached {MAX_ITEMS} items. Stopping.")
                        return

                    issuer_name = issuer_obj.get("issuerName")
                    if not issuer_name:
                        continue

                    item = {
                        "issuerName":      issuer_name,                   # PK
                        "publicationDate": pub_date,                      # SK
                        "reportId":        str(report_id) if report_id is not None else None,
                        "title":           rec.get("title"),
                        "url":             rec.get("url"),
                        "isPriority":      rec.get("isPriorityReport", False),
                        "isCorrection":    rec.get("isCorrection", False),
                    }

                    if events_map:
                        item["events"] = events_map
                    if s3_keys:
                        item["attachedFiles"] = s3_keys

                    # Drop reportId if it's None
                    if item.get("reportId") is None:
                        item.pop("reportId", None)

                    try:
                        table.put_item(Item=item)
                        inserted_count += 1
                        print(f"→ ({inserted_count}/{MAX_ITEMS}) Wrote issuer={issuer_name}@{pub_date}")
                    except ClientError as e:
                        print(f"   ✖ DynamoDB put_item failed for issuer={issuer_name}@{pub_date}: {e}")

    print(f"✔ Finished scanning all JSONs. Total items inserted: {inserted_count}")


if __name__ == "__main__":
    main()
