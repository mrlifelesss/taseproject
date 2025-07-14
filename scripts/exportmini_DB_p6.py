#!/usr/bin/env python3
# exportmini_DB_p6_with_full_count.py
#
# Multithreaded version **with a paginated full-table scan** for current_count()
# ──────────────────────────────────────────────────────────────────────────────

import os, json, urllib.parse, mimetypes, requests, threading, concurrent.futures
import boto3
from botocore.exceptions import ClientError

# ─── CONFIG ───────────────────────────────────────────────────────────
AWS_REGION          = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TABLE_NAME          = "CompanyDisclosuresHebrew"
SOURCE_TABLE_NAME   = "CompanyDisclosures"
S3_BUCKET           = "summery-prompts-data"
JSON_PREFIX         = "apicallsdata/"
ATTACH_PREFIX       = "apicallsdata/attachedfiles"
MAX_TABLE_SIZE      = 20_000
EVENT_ID_MAX        = 5000
REPORT_ID_THRESHOLD = 0
MAX_WORKERS         = (os.cpu_count() or 4) * 5   # tweak to taste
ADD_DUMMY           = True
DUMMY_FIELD_NAME    = "dummy"
DUMMY_FIELD_VALUE   = "True"

# ─── AWS CLIENTS (shared — thread-safe) ───────────────────────────────
session   = boto3.Session(region_name=AWS_REGION)
ddb       = session.resource("dynamodb")
s3        = session.client("s3")
table     = ddb.Table(TABLE_NAME)
source    = ddb.Table(SOURCE_TABLE_NAME)
paginator = s3.get_paginator("list_objects_v2")

# ─── HELPERS ──────────────────────────────────────────────────────────
def purge_items_with_k():
    "Delete every row whose form_type contains Hebrew 'ק', in bulk."
    print("➤ Purging items whose form_type contains 'ק' …")
    deleted = scanned = 0
    last = None
    while True:
        scan_kwargs = {
            "ProjectionExpression": "issuerName,publicationDate,form_type,reportId"
        }
        if last:
            scan_kwargs["ExclusiveStartKey"] = last

        resp = table.scan(**scan_kwargs)
        with table.batch_writer(overwrite_by_pkeys=("issuerName", "publicationDate")) as bw:
            for item in resp.get("Items", []):
                scanned += 1
                ftype = item.get("form_type") or _lookup_form_type(item)
                if ftype and "ק" in ftype:
                    bw.delete_item(
                        Key={
                            "issuerName":      item["issuerName"],
                            "publicationDate": item["publicationDate"],
                        }
                    )
                    deleted += 1

        last = resp.get("LastEvaluatedKey")
        if not last:
            break

    print(f"✔ Purge done. Scanned {scanned}, deleted {deleted}.")


def _lookup_form_type(item: dict) -> str | None:
    "Helper: get form_type from the source table if missing."
    try:
        src = source.get_item(
            Key={
                "reportId":        item["reportId"],
                "publicationDate": item["publicationDate"],
            }
        ).get("Item")
        return src.get("form_type") if src else None
    except ClientError:
        return None


def record_has_bad_event(rec: dict) -> bool:
    """
    Return True if *any* eventId inside rec['events'] exceeds EVENT_ID_MAX.
    """
    for ev in rec.get("events", []):
        try:
            if int(ev.get("eventId", 0)) > EVENT_ID_MAX:
                return True
        except ValueError:
            continue
    return False


def enrich_from_source(report_id: str, pubdate: str) -> dict:
    "Return the 7 enrichment fields (or {} if anything goes wrong)."
    try:
        resp = source.get_item(Key={"reportId": report_id, "publicationDate": pubdate})
        src = resp.get("Item") or {}
        want = (
            "language",
            "pdf_count",
            "pdf_total_size",
            "proof_id",
            "registration_number",
            "subject",
            "form_type",
        )
        return {k: src[k] for k in want if k in src}
    except ClientError:
        return {}


def upload_attachments(report_id: str, urls: list[str]) -> list[str]:
    """
    Download each URL and re-upload into S3; return a list of the new S3 keys.
    (If any download/upload fails, skip that attachment silently.)
    """
    keys = []
    for url in urls:
        if not url:
            continue
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
        except Exception:
            continue

        fname = os.path.basename(urllib.parse.urlparse(url).path)
        key = f"{ATTACH_PREFIX}/{report_id}/{fname}"
        ctype = (
            r.headers.get("Content-Type")
            or mimetypes.guess_type(fname)[0]
            or "application/octet-stream"
        )

        try:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=r.content, ContentType=ctype)
            keys.append(key)
        except ClientError:
            pass

    return keys


def current_count() -> int:
    """
    Perform a paginated Scan(Select="COUNT") until all pages are read.
    This returns the *true* number of items in the table, not just the first 1 MB page.
    """
    total = 0
    resp = table.scan(Select="COUNT")
    total += resp.get("Count", 0)

    # Keep scanning more pages until DynamoDB indicates there are no more
    while "LastEvaluatedKey" in resp:
        resp = table.scan(
            Select="COUNT", ExclusiveStartKey=resp["LastEvaluatedKey"]
        )
        total += resp.get("Count", 0)

    return total


# ─── MULTITHREADED INGEST ────────────────────────────────────────────
inserted = 0
insert_lock = threading.Lock()
stop_event = threading.Event()
start_items = 0  # will be set right before ingest phase


def worker(rec: dict, issuer_name: str, pubdate: str):
    """
    Each worker does:
      1) skip if any eventId > EVENT_ID_MAX
      2) download & re-upload attachments
      3) build the DynamoDB item
      4) put_item(...) into the table
      5) increment the inserted counter (thread-safe)
    """
    global inserted

    if stop_event.is_set():
        return

    # 1. If any eventId in this record exceeds EVENT_ID_MAX, skip entirely
    if record_has_bad_event(rec):
        return

    report_id = str(rec.get("mayaReportId"))
    if not report_id:
        return

    # 2. Download + re-upload attachments
    attach_urls = [a.get("url", "").strip() for a in rec.get("attachedfiles", [])]
    s3_keys = upload_attachments(report_id, attach_urls)

    # 3. Build the DynamoDB item
    item = {
        "issuerName":      issuer_name,
        "publicationDate": pubdate,
        "reportId":        report_id,
        "title":           rec.get("title"),
        "url":             rec.get("url"),
        "isPriority":      rec.get("isPriorityReport", False),
        "isCorrection":    rec.get("isCorrection", False),
    }

    if rec.get("events"):
        # Build a map of eventName → eventObject
        item["events"] = {
            ev["eventName"]: ev
            for ev in rec["events"]
            if ev.get("eventName")
        }

    if s3_keys:
        item["attachedFiles"] = s3_keys

    if ADD_DUMMY:
        item[DUMMY_FIELD_NAME] = DUMMY_FIELD_VALUE

    # 4. Enrich with the 7 extra fields from the source table
    item.update(enrich_from_source(report_id, pubdate))

    try:
        table.put_item(Item=item)
    except ClientError:
        # A transient DynamoDB error? Just skip
        return

    # 5. Increment the shared counter in a thread-safe block
    with insert_lock:
        inserted += 1
        remaining = MAX_TABLE_SIZE - start_items
        if inserted >= remaining:
            stop_event.set()
        elif inserted % 200 == 0:
            print(f"   • inserted {inserted}/{remaining}")


def ingest_until_target_mt():
    """
    1) Read the *true* current item count (using paginated scan).
    2) If count < MAX_TABLE_SIZE, start paginating through the S3 JSON prefix.
    3) For each valid record/issuer, submit a worker(...) to ThreadPool.
    4) Exit once we've inserted enough items or exhausted all JSONs.
    """
    global start_items
    start_items = current_count()
    print(f"➤ Table currently holds ~{start_items} items.")
    if start_items >= MAX_TABLE_SIZE:
        print("✓ Already at or above target; nothing to insert.")
        return

    to_insert = MAX_TABLE_SIZE - start_items

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=JSON_PREFIX):
            if stop_event.is_set():
                break

            for obj in page.get("Contents", []):
                if stop_event.is_set():
                    break

                key = obj["Key"]
                if not key.lower().endswith(".json"):
                    continue

                # Fetch each JSON and parse it
                try:
                    raw = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                    data = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    print(f"  [Skipping {key}] failed to load/parse: {e}")
                    continue

                for rec in data.get("mayaReports", {}).get("result", []):
                    if stop_event.is_set():
                        break
                    try:
                        prid = int(rec.get("mayaReportId", 0))
                        pubdate = rec.get("publicationDate")
                    except (ValueError, TypeError):
                        continue

                    if (prid <= REPORT_ID_THRESHOLD) or (not pubdate):
                        continue

                    for issuer in rec.get("issuer", []):
                        if stop_event.is_set():
                            break

                        issuer_name = issuer.get("issuerName")
                        if not issuer_name:
                            continue

                        pool.submit(worker, rec, issuer_name, pubdate)

        # Wait for all worker threads to finish (or for stop_event to fire)
        pool.shutdown(wait=True)

    print(
        f"✔ Finished ingest: inserted {inserted} items "
        f"(table should now be ~{start_items + inserted})."
    )


# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    # If you still want to purge first, uncomment the next line:
    # purge_items_with_k()

    ingest_until_target_mt()
    print("🏁 All done!")


if __name__ == "__main__":
    main()
