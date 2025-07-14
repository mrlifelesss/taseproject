import os
import boto3
from botocore.exceptions import ClientError

# ─── CONFIG ────────────────────────────────────────────────────────────────
AWS_REGION          = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TABLE_NAME          = "CompanyDisclosuresHebrew"   # Your existing table
DUMMY_FIELD_NAME    = "dummy"
DUMMY_FIELD_VALUE   = "True"                       # As a string; change to True (no quotes) for a boolean

# ─── SET UP DYNAMODB RESOURCE ──────────────────────────────────────────────
session = boto3.Session(region_name=AWS_REGION)
ddb     = session.resource("dynamodb")
table   = ddb.Table(TABLE_NAME)

def add_dummy_to_all_items():
    """
    Scans through every item in CompanyDisclosuresHebrew and adds
    a new attribute "dummy"="True" (or boolean True if you prefer).
    """
    last_evaluated_key = None
    updated_count = 0
    scanned_count = 0

    while True:
        scan_kwargs = {}
        if last_evaluated_key:
            scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

        try:
            response = table.scan(**scan_kwargs)
        except ClientError as e:
            print(f"✖ Error scanning {TABLE_NAME}: {e}")
            return

        items = response.get("Items", [])
        for item in items:
            scanned_count += 1

            # Extract the primary key fields from each item
            pk_issuer       = item.get("issuerName")
            sk_pubdate      = item.get("publicationDate")

            if not pk_issuer or not sk_pubdate:
                # If either key is missing, skip
                continue

            try:
                table.update_item(
                    Key={
                        "issuerName":      pk_issuer,
                        "publicationDate": sk_pubdate
                    },
                    UpdateExpression=f"SET #{DUMMY_FIELD_NAME} = :val",
                    ExpressionAttributeNames={
                        f"#{DUMMY_FIELD_NAME}": DUMMY_FIELD_NAME
                    },
                    ExpressionAttributeValues={
                        ":val": DUMMY_FIELD_VALUE
                    }
                )
                updated_count += 1
                if updated_count % 100 == 0:
                    print(f"→ Updated {updated_count} items so far...")
            except ClientError as e:
                print(f"   ✖ Failed to update {pk_issuer}@{sk_pubdate}: {e}")

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    print(f"\n✔ Scan complete. Scanned {scanned_count} items; updated {updated_count} items "
          f"with {DUMMY_FIELD_NAME}='{DUMMY_FIELD_VALUE}'.")


if __name__ == "__main__":
    add_dummy_to_all_items()
