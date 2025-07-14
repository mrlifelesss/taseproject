import json
import pandas as pd

# Load JSON data
with open('latest_disclosures.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract the list of reports
records = data.get('mayaReports', {}).get('result', [])

# Flatten nested structures into a list of simple dicts
flat_records = []
for r in records:
    flat_records.append({
        'publicationDate': r.get('publicationDate'),
        'mayaReportId': r.get('mayaReportId'),
        'isPriorityReport': r.get('isPriorityReport'),
        'isCorrection': r.get('isCorrection'),
        'title': r.get('title'),
        'url': r.get('url'),
        'issuers': ', '.join(issuer.get('issuerName', '') for issuer in r.get('issuer', [])),
        'events': ', '.join(event.get('eventName', '') for event in r.get('events', [])),
        'attachedUrls': ', '.join(file.get('url', '') for file in r.get('attachedfiles', [])),
    })

# Create DataFrame
df = pd.DataFrame(flat_records)

# Write to Excel
out_path = 'latest_disclosures_flat.xlsx'
df.to_excel(out_path, index=False)


print(f"✅ File saved: {out_path}")
