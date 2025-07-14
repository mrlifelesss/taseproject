#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key

def main():
    p = argparse.ArgumentParser(
        description="For each doc_id in CSV, fetch its HTML from DynamoDB/S3 and run clean_text_test.py"
    )
    p.add_argument("--csv", "-c", default="doc_vectors2.csv",
                   help="CSV file with header column 'doc_id'")
    p.add_argument("--table", "-t", default="CompanyDisclosuresHebrew",
                   help="DynamoDB table name")
    p.add_argument("--index", "-i", default="reportId-index",
                   help="GSI name on reportId")
    p.add_argument("--bucket", "-b", default="summery-prompts-data",
                   help="S3 bucket for attachedFiles")
    p.add_argument("--region", "-r", default="us-east-1",
                   help="AWS region")
    p.add_argument("--out", "-o", default="cleanHTM",
                   help="Output directory for cleaned .txt files")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("tmp_html")
    tmp_dir.mkdir(exist_ok=True)

    # AWS clients
    dynamo = boto3.resource("dynamodb", region_name=args.region)
    table = dynamo.Table(args.table)
    s3 = boto3.client("s3", region_name=args.region)

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id") or row.get("reportId")
            if not doc_id:
                print(f"⚠️  skipping row without doc_id: {row}")
                continue

            # query GSI
            resp = table.query(
                IndexName=args.index,
                KeyConditionExpression=Key("reportId").eq(doc_id)
            )
            items = resp.get("Items", [])
            if not items:
                print(f"⚠️  no DynamoDB item for reportId={doc_id}")
                continue

            # take the first item's attachedFiles
            files = items[0].get("attachedFiles", [])
            if not files:
                print(f"⚠️  no attachedFiles for reportId={doc_id}")
                continue

            key = files[0].lstrip("/")
            local_html = tmp_dir / f"{doc_id}.html"
            try:
                obj = s3.get_object(Bucket=args.bucket, Key=key)
                html = obj["Body"].read()
                local_html.write_bytes(html)
            except Exception as e:
                print(f"[ERR] downloading {key}: {e}")
                continue

            out_txt = out_dir / f"{doc_id}.txt"
            # call clean_text_test.py
            cmd = [
                sys.executable,
                "scripts/clean_text_test.py",
                str(local_html),
                str(out_txt)
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERR] cleaning {doc_id}: exit {e.returncode}")
            else:
                print(f"✅  {doc_id} → {out_txt}")

if __name__ == "__main__":
    main()
