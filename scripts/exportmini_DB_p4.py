import os
import json
import boto3
from botocore.exceptions import ClientError
from decimal import Decimal
# ─── CONFIG ────────────────────────────────────────────────────────────────
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TABLE_NAME   = "CompanyDisclosuresHebrew"
PAGE_SIZE    = 100     # number of items to scan per page

# ─── DYNAMODB SETUP ─────────────────────────────────────────────────────────
ddb   = boto3.resource("dynamodb", region_name=AWS_REGION)
table = ddb.Table(TABLE_NAME)

def scan_all_items(page_size=PAGE_SIZE):
    """
    Generator that paginates through the entire table and yields each item.
    """
    last_evaluated_key = None
    while True:
        scan_kwargs = {"Limit": page_size}
        if last_evaluated_key:
            scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

        try:
            response = table.scan(**scan_kwargs)
        except ClientError as e:
            print(f"✖ Error scanning {TABLE_NAME}: {e}")
            return

        for item in response.get("Items", []):
            yield item

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break


def convert_decimals(obj):
    """
    Recursively walk a Python object (which may be a dict, list, Decimal, or primitive)
    and return a new structure where every Decimal is converted to int.
    """
    if isinstance(obj, list):
        return [convert_decimals(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        # If you know these are always “integers,” cast to int. Otherwise, float(obj) is also okay.
        try:
            return int(obj)
        except (ValueError, OverflowError):
            return float(obj)
    # For any other type (str, int, float, bool, None), leave as-is
    return obj

def stringify_events(item):
    """
    Take the existing 'events' attribute (anything except a plain string),
    convert all Decimal → int, then JSON‐serialize it.
    If 'events' is missing or already a string, return None.
    """
    if "events" not in item:
        return None

    current = item["events"]

    # If it's already a Python str, skip it (we assume it’s been stringified already).
    if isinstance(current, str):
        return None

    # First, convert any Decimal in the structure to int/float
    clean_obj = convert_decimals(current)

    # Now JSON‐dump with ensure_ascii=False so Hebrew stays readable
    try:
        new_string = json.dumps(clean_obj, ensure_ascii=False)
        return new_string
    except (TypeError, ValueError) as e:
        print(f"⚠ Skipping item {item.get('issuerName')}@{item.get('publicationDate')}: "
              f"could not JSON‐serialize events even after cleaning: {e}")
        return None


def update_item_events_to_string(pk_issuer, sk_pubdate, json_str):
    """
    Update the item identified by (issuerName=pk_issuer, publicationDate=sk_pubdate),
    setting its 'events' attribute to the provided JSON string.
    """
    try:
        table.update_item(
            Key={
                "issuerName":      pk_issuer,
                "publicationDate": sk_pubdate
            },
            UpdateExpression="SET #ev = :val",
            ExpressionAttributeNames={"#ev": "events"},
            ExpressionAttributeValues={":val": json_str}
        )
        print(f"✔ Converted events to string for {pk_issuer}@{sk_pubdate}")
    except ClientError as e:
        print(f"   ✖ Failed to update {pk_issuer}@{sk_pubdate}: {e}")


def main():
    scanned = 0
    updated = 0

    for item in scan_all_items():
        scanned += 1

        pk_issuer  = item.get("issuerName")
        sk_pubdate = item.get("publicationDate")

        if not pk_issuer or not sk_pubdate:
            # Skip any item missing the required keys
            continue

        new_str = stringify_events(item)
        if new_str is None:
            # No need to change (either no 'events' or already a string)
            continue

        update_item_events_to_string(pk_issuer, sk_pubdate, new_str)
        updated += 1

    print(f"\n--- Done ---\nScanned {scanned} items; updated {updated} items.")


if __name__ == "__main__":
    main()
