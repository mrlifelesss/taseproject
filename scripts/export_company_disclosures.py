#!/usr/bin/env python3
"""
Export selected fields from CompanyDisclosures DynamoDB table to Excel.
"""

import os
import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TABLE_NAME  = os.environ.get('DDB_TABLE', 'CompanyDisclosures')
OUTPUT_FILE = 'disclosures_summary.xlsx'
# ────────────────────────────────────────────────────────────────────────────────

def scan_table(table):
    """Scan entire table (with pagination) and yield each item."""
    paginator = table.meta.client.get_paginator('scan')
    scan_kwargs = {
        'TableName': TABLE_NAME,
        # Only pull items that already have our enriched fields
        'FilterExpression': Attr('form_type').exists() &
                            Attr('language').exists() &
                            Attr('pdf_count').exists() &
                            Attr('pdf_total_size').exists(),
        'ProjectionExpression': 'reportId, form_type, #lang, pdf_count, pdf_total_size',
        'ExpressionAttributeNames': {
            '#lang': 'language'
        }
    }

    for page in paginator.paginate(**scan_kwargs):
        for it in page['Items']:
            yield it

def main():
    # init
    dynamodb = boto3.resource('dynamodb')
    table     = dynamodb.Table(TABLE_NAME)

    # collect rows
    rows = []
    for item in scan_table(table):
        rows.append({
            'ID':               item.get('reportId'),
            'Type':             item.get('form_type'),
            'Language':         item.get('language'),
            'PDF Count':        item.get('pdf_count'),
            'PDF Total Size':   item.get('pdf_total_size'),
        })

    if not rows:
        print("No items found matching criteria.")
        return

    # build DataFrame and write Excel
    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
