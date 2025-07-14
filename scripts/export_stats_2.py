#!/usr/bin/env python3
"""
export_stats_handle_events_str.py

Scan CompanyDisclosuresHebrew, parse the JSON‐string in `events`, “explode” each item
by its top‐level event keys, then group by (form_type, event_name) to compute:
   • total_reports
   • avg_pdfs
   • avg_pdf_size_kb

Writes form_type_event_stats.xlsx with one row per (form_type, event_name),
and prints both the total number of items DynamoDB scanned (ScannedCount) as well
as the number of items that matched our FilterExpression.
"""

import os
import json
import boto3
import pandas as pd
from collections import defaultdict
from boto3.dynamodb.conditions import Attr
from decimal import Decimal

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TABLE_NAME  = os.environ.get('DDB_TABLE', 'CompanyDisclosuresHebrew')
OUTPUT_FILE = 'form_type_event_stats.xlsx'
# ────────────────────────────────────────────────────────────────────────────────

def scan_table_and_count():
    """
    Scan the entire table with a filter that requires:
      - form_type
      - pdf_count
      - pdf_total_size
      - events

    Yields a tuple (items_on_this_page, scanned_count_for_this_page). 
    - items_on_this_page is a Python list of all items that passed the filter on that page.
    - scanned_count_for_this_page is how many items DynamoDB internally scanned on that page.
    """
    dynamodb = boto3.resource('dynamodb')
    table    = dynamodb.Table(TABLE_NAME)

    # Build a FilterExpression requiring that all four fields exist
    filter_expr = (
        Attr('form_type').exists()
        & Attr('pdf_count').exists()
        & Attr('pdf_total_size').exists()
        & Attr('events').exists()
    )

    # We only project the four fields we need
    scan_kwargs = {
        'FilterExpression': filter_expr,
        'ProjectionExpression': 'form_type, pdf_count, pdf_total_size, events'
    }

    while True:
        page = table.scan(**scan_kwargs)

        # How many items DynamoDB physically scanned on this page
        scanned_count_for_this_page = page.get('ScannedCount', 0)
        # Which items passed the filter? 
        items_on_this_page = page.get('Items', [])

        yield items_on_this_page, scanned_count_for_this_page

        # If there’s no LastEvaluatedKey, we’re done
        if 'LastEvaluatedKey' not in page:
            break

        # Otherwise, start the next scan after this key
        scan_kwargs['ExclusiveStartKey'] = page['LastEvaluatedKey']


def accumulate_stats():
    """
    Iterate over scanned pages, parse each page’s items, “explode” by event key, etc.
    Keep two separate counters:
      1. total_scanned_count  = sum of ScannedCount from each page (added exactly once per page)
      2. matched_count        = number of items that actually passed the filter

    Returns a dict mapping (form_type, event_name) → {count, sum_pdfs, sum_bytes}.
    """
    stats = defaultdict(lambda: {'count': 0, 'sum_pdfs': 0, 'sum_bytes': 0})
    total_scanned_count = 0
    matched_count       = 0

    for items_on_page, page_scanned_count in scan_table_and_count():
        # Add the scanned count once for this entire page
        total_scanned_count += page_scanned_count

        # Process each item that passed the filter on this page
        for raw_item in items_on_page:
            matched_count += 1

            form_type = raw_item.get('form_type')

            # Convert pdf_count (which may be a Decimal) into int
            pdf_count = raw_item.get('pdf_count', 0)
            if isinstance(pdf_count, Decimal):
                pdf_count = int(pdf_count)
            else:
                pdf_count = int(pdf_count)

            # Convert pdf_total_size (which may be a Decimal) into int
            total_bytes = raw_item.get('pdf_total_size', 0)
            if isinstance(total_bytes, Decimal):
                total_bytes = int(total_bytes)
            else:
                total_bytes = int(total_bytes)

            raw_events = raw_item.get('events', "")
            if not isinstance(raw_events, str) or not raw_events.strip():
                # Skip if events isn’t a non-empty string
                continue

            try:
                events_map = json.loads(raw_events)
            except json.JSONDecodeError:
                # Skip malformed JSON
                continue

            if not isinstance(events_map, dict) or len(events_map) == 0:
                # Skip if it’s not a non-empty dict
                continue

            # “Explode” out each event_name under this item
            for event_name in events_map.keys():
                key = (form_type, event_name)
                stats[key]['count']    += 1
                stats[key]['sum_pdfs'] += pdf_count
                stats[key]['sum_bytes'] += total_bytes

    # After scanning all pages, print totals:
    print(f"Scanned {total_scanned_count:,} items in total (DynamoDB ‘ScannedCount’).")
    print(f"Matched  {matched_count:,} items where form_type/pdf_count/pdf_total_size/events all exist.\n")

    return stats


def build_dataframe(stats: dict) -> pd.DataFrame:
    """
    Given the stats dict from accumulate_stats(), produce a pandas DataFrame with columns:
      form_type, event_name, total_reports, avg_pdfs, avg_pdf_size_kb
    Averages are rounded to 3 decimals, trailing zeros dropped.
    """
    rows = []
    for (form_type, event_name), vals in stats.items():
        n = vals['count']
        if n == 0:
            continue

        avg_pdfs    = vals['sum_pdfs'] / n
        avg_size_kb = (vals['sum_bytes'] / n) / 1024  # convert bytes → KB

        # Format to 3 decimal places, then drop trailing zeros
        avg_pdfs_str    = f"{round(avg_pdfs, 3):.3f}".rstrip('0').rstrip('.')
        avg_size_kb_str = f"{round(avg_size_kb, 3):.3f}".rstrip('0').rstrip('.')

        rows.append({
            'form_type':       form_type,
            'event_name':      event_name,
            'total_reports':   n,
            'avg_pdfs':        avg_pdfs_str,
            'avg_pdf_size_kb': avg_size_kb_str
        })

    df = pd.DataFrame(rows, columns=[
        'form_type', 'event_name', 'total_reports', 'avg_pdfs', 'avg_pdf_size_kb'
    ])
    return df.sort_values(by='total_reports', ascending=False)


def main():
    stats = accumulate_stats()
    if not stats:
        print("No data found (or no items with form_type/pdf_count/pdf_total_size/events).")
        return

    df = build_dataframe(stats)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
