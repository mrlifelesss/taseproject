import os
import boto3
from botocore.exceptions import ClientError

# ─── CONFIG ────────────────────────────────────────────────────────────────
AWS_REGION                = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
HEBREW_TABLE_NAME         = "CompanyDisclosuresHebrew"   # PK: issuerName (S), SK: publicationDate (S)
SOURCE_TABLE_NAME         = "CompanyDisclosures"         # PK: reportId (S)
PAGE_SIZE                 = 100                         # How many items to retrieve per scan page

# ─── SET UP DYNAMODB RESOURCE & TABLE OBJECTS ──────────────────────────────
session = boto3.Session(region_name=AWS_REGION)
ddb     = session.resource("dynamodb")

hebrew_table = ddb.Table(HEBREW_TABLE_NAME)
source_table = ddb.Table(SOURCE_TABLE_NAME)

def scan_hebrew_table():
    """
    Generator that scans CompanyDisclosuresHebrew in pages of PAGE_SIZE,
    yielding each item as a Python dict.
    """
    scan_kwargs = {
        "Limit": PAGE_SIZE
    }
    last_key = None

    while True:
        if last_key:
            scan_kwargs["ExclusiveStartKey"] = last_key

        try:
            response = hebrew_table.scan(**scan_kwargs)
        except ClientError as e:
            print(f"✖ Error scanning {HEBREW_TABLE_NAME}: {e}")
            return

        for item in response.get("Items", []):
            yield item

        last_key = response.get("LastEvaluatedKey", None)
        if not last_key:
            break

def fetch_source_fields(report_id, publication_date):
    """
    Given a report_id (String) and its publication_date (String),
    retrieve that same item from CompanyDisclosures, which has:
      • PK = reportId (String)
      • SK = publicationDate (String)
    """
    try:
        resp = source_table.get_item(
            Key={
                "reportId":       report_id,         # must be a Python str
                "publicationDate": publication_date # also a Python str
            }
        )
    except ClientError as e:
        print(f"   ✖ Error retrieving reportId={report_id} @ {publication_date} from CompanyDisclosures: {e}")
        return None

    src_item = resp.get("Item")
    if not src_item:
        # no matching item in the source table
        return None

    # Only pull the six fields you care about:
    fields = {}
    for fld in (
        "language",
        "pdf_count",
        "pdf_total_size",
        "proof_id",
        "registration_number",
        "subject",
        "form_type"
    ):
        if fld in src_item:
            fields[fld] = src_item[fld]

    return fields or None


def update_hebrew_item(pk_issuer, sk_pubdate, new_fields):
    """
    Updates a single item in CompanyDisclosuresHebrew identified by
    (issuerName=pk_issuer, publicationDate=sk_pubdate), adding new_fields.
    """
    # Build UpdateExpression, ExpressionAttributeNames, and ExpressionAttributeValues
    #
    # e.g.  "SET #lang = :language, #pdfCount = :pdf_count, ... "
    update_expr_parts = []
    expr_attr_names  = {}
    expr_attr_values = {}

    for i, (field, val) in enumerate(new_fields.items()):
        placeholder_name  = f"#f{i}"
        placeholder_value = f":v{i}"
        update_expr_parts.append(f"{placeholder_name} = {placeholder_value}")
        expr_attr_names[placeholder_name]  = field
        expr_attr_values[placeholder_value] = val

    update_expression = "SET " + ", ".join(update_expr_parts)

    try:
        hebrew_table.update_item(
            Key={
                "issuerName":      pk_issuer,
                "publicationDate": sk_pubdate
            },
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expr_attr_names,
            ExpressionAttributeValues=expr_attr_values
        )
        print(f"✔ Updated issuer={pk_issuer}@{sk_pubdate} with {list(new_fields.keys())}")
    except ClientError as e:
        print(f"   ✖ Failed to update issuer={pk_issuer}@{sk_pubdate}: {e}")

def main():
    scanned = 0
    updated = 0

    for hebrew_item in scan_hebrew_table():
        scanned += 1
        report_id       = hebrew_item.get("reportId")
        publicationDate = hebrew_item.get("publicationDate")

        # Must supply both PK and SK to get the source‐table item
        if not report_id or not publicationDate:
            continue

        new_fields = fetch_source_fields(report_id, publicationDate)
        if not new_fields:
            continue

        update_hebrew_item(
            hebrew_item["issuerName"],
            publicationDate,
            new_fields
        )
        updated += 1

    print(f"Scanned {scanned} items; updated {updated} items.")

if __name__ == "__main__":
    main()