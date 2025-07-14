import http.client
import os
import json
import csv
import pandas as pd
import boto3
import datetime
import time

API_KEY = os.getenv("TASE_API_KEY")

# Verify API key is loaded
if not API_KEY:
    raise ValueError("TASE_API_KEY environment variable is not set")

headers = {
    'accept': "application/json",
    'accept-language': "he-IL",
    'apikey': API_KEY
}

BASE_PATH = "/v1/maya-reports-online/companies-disclosures-by-date"

# Verify BASE_PATH is properly set
if not BASE_PATH:
    raise ValueError("BASE_PATH is not properly defined")

# Setup S3 client once
bucket = "summery-prompts-data"
prefix = "apicallsdata"  # folder inside the bucket

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# Loop over the last 460 days (changed from 461 to match your comment about 90 days)
today = datetime.date(2022, 1, 2)  # e.g. 2025-05-22

for delta in range(1, 2):  # or change to 91 if you want 90 days
    d = today - datetime.timedelta(days=delta)
    
    # Construct path with error checking
    path = f"{BASE_PATH}/{d.year}/{d.month}/{d.day}"
    
    # Debug print to verify path construction
    print(f"→ Fetching {d.isoformat()} (path: {path}) …", end=" ")
    
    # Create a new connection for each request to avoid connection issues
    conn = http.client.HTTPSConnection("datawise.tase.co.il")
    
    try:
        # Verify path is a string and not None
        if path is None:
            print(f"❌ Path is None for date {d.isoformat()}")
            continue
            
        conn.request("GET", path, headers=headers)
        res = conn.getresponse()
        raw = res.read()  # keeps it as bytes

        if res.status == 200:
            key = f"{prefix}/{d.isoformat()}.json"
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=raw,
                ContentType="application/json",
                Tagging="Round 2"
            )
            print(f"✔️ Uploaded to s3://{bucket}/{key}")
        else:
            print(f"❌ HTTP {res.status}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Close the connection
        conn.close()

    # Small pause to be gentle on the API
    time.sleep(0.1)