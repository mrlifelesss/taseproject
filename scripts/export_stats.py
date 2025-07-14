#!/usr/bin/env python3
import boto3
from collections import defaultdict
import pandas as pd
from boto3.dynamodb.conditions import Attr

# ─── CONFIG ──────────────────────────────────────────────────────
TABLE_NAME = 'CompanyDisclosuresHebrew'
OUTPUT_XLSX = 'disclosure_stats_mini.xlsx'
# ──────────────────────────────────────────────────────────────────

dynamodb = boto3.resource('dynamodb')
table     = dynamodb.Table('CompanyDisclosures')

# 1) Pull only the fields we need
paginator = table.meta.client.get_paginator('scan')
scan_kwargs = {
    'TableName': TABLE_NAME,
    'FilterExpression': Attr('form_type').exists() & Attr('language').exists(),
    'ProjectionExpression': 'form_type, #lang, pdf_count, pdf_total_size',
    'ExpressionAttributeNames': {
            '#lang': 'language'
        }
}

stats = defaultdict(lambda: {'count':0,'sum_pdfs':0,'sum_size':0})
for page in paginator.paginate(**scan_kwargs):
    for it in page['Items']:
        ft, lang = it['form_type'], it['language']
        key = (ft,lang)
        stats[key]['count']    += 1
        stats[key]['sum_pdfs'] += it.get('pdf_count',0)
        stats[key]['sum_size'] += it.get('pdf_total_size',0)

# 2) Turn it into a list of rows
rows = []
for (ft, lang), v in stats.items():
    n   = v['count']
    rows.append({
        'form_type':        ft,
        'language':         lang,
        'total_reports':    n,
        'avg_pdfs':         round(v['sum_pdfs'] / n, 3),
        'avg_pdf_size':     round(v['sum_size'] / n, 3)
    })

# 3) Build DataFrame & write Excel
df = pd.DataFrame(rows)
# 3.1) convert bytes → KB
df['avg_pdf_size'] = df['avg_pdf_size'] / 1024

# 3.2) round to 3 decimal places and  drop trailing zeros and any trailing decimal point
df['avg_pdf_size'] = df['avg_pdf_size']\
    .round(3)\
    .apply(lambda x: f"{x:.3f}".rstrip('0').rstrip('.'))

df['avg_pdfs'] = df['avg_pdfs']\
    .round(3)\
    .apply(lambda x: f"{x:.3f}".rstrip('0').rstrip('.'))
# 3.3) rename the column to indicate KB
df = df.rename(columns={'avg_pdf_size': 'avg_pdf_size_kb'})
# 4) sort ascending by the number of reports
df = df.sort_values(by='total_reports', ascending=False)

df.to_excel(OUTPUT_XLSX, index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_XLSX}")
