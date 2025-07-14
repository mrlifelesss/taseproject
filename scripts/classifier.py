import json
import re
import os
import argparse

# --- Keywords for Stage 2 Classification ---
CORP_IR_KEYWORDS = ['מצגת משקיעים', 'שיחת ועידה', 'מועד פרסום דוחות', 'כנס משקיעים', 'Zoom']
CORP_MATERIAL_EVENTS_KEYWORDS = ['תובענה ייצוגית', 'פסק דין', 'הסכם מהותי', 'זכייה במכרז', 'אישור רגולטורי']
CORP_HOLDER_MEETINGS_KEYWORDS = ['אסיפה כללית', 'סדר יום', 'כתב הצבעה']
CANDIDATE_PROSPECTUS_ISSUANCE_KEYWORDS = ['תשקיף מדף', 'הצעה פרטית']
FIN_CAPITAL_STRUCTURE_KEYWORDS = ['דיבידנד', 'מניות רדומות', 'רכישה עצמית']


def extract_form_symbol(text: str) -> str | None:
    """
    Extracts the form symbol (e.g., 'ת121') from the beginning of the text.
    Searches only the first 200 characters for efficiency.
    """
    if not text:
        return None
    match = re.search(r'(ת|א|ד)\d{3}', text[:200])
    return match.group(0) if match else None


def normalize_event_ids(event_ids) -> str:
    """
    Convert eventIds field to comma-separated string.
    """
    if isinstance(event_ids, str):
        return event_ids
    if isinstance(event_ids, (int, float)):
        return str(event_ids)
    if isinstance(event_ids, list):
        return ','.join(str(e) for e in event_ids)
    return ''


def classify_document(form_type: str, event_ids_str: str, full_text: str) -> str:
    """
    Classifies a document based on the provided logic.
    """
    event_ids = event_ids_str.split(',') if event_ids_str else []
    # Stage 1
    if form_type in ['ת076', 'ת077', 'ת078']:
        return 'CORP_INSIDER_HOLDINGS'
    if form_type in ['ת094', 'ת091', 'ת093', 'ת097', 'ת304', 'ת306']:
        return 'MGMT_OFFICER_CHANGE'
    if form_type in ['ת081', 'ת085', 'ת086', 'ת087', 'ת310', 'ת880']:
        return 'FIN_CAPITAL_STRUCTURE'
    if form_type in ['ת126', 'ת930']:
        return 'FIN_FINANCIAL_REPORTS'
    if form_type in ['ת100', 'ת101']:
        return 'CORP_ADMIN_UPDATE'
    if form_type in ['ד125', 'ת125']:
        return 'CANDIDATE_CREDIT_RATING'
    if form_type == 'ת089':
        return 'CANDIDATE_INTEREST_PAYMENTS'
    if form_type == 'א121':
        return 'NEW_CANDIDATE_BOND_TRUSTEE_CHANGE'

    # Stage 2
    if form_type in ['ת121', 'ת053'] or '201' in event_ids:
        if any(k in full_text for k in CORP_IR_KEYWORDS):
            return 'CORP_INVESTOR_RELATIONS'
        if any(k in full_text for k in CORP_MATERIAL_EVENTS_KEYWORDS):
            return 'CORP_MATERIAL_EVENTS'
    if '602' in event_ids:
        if any(k in full_text for k in CORP_HOLDER_MEETINGS_KEYWORDS):
            return 'CORP_HOLDER_MEETINGS'
        return 'MGMT_OFFICER_CHANGE'
    if '707' in event_ids:
        if any(k in full_text for k in CANDIDATE_PROSPECTUS_ISSUANCE_KEYWORDS):
            return 'CANDIDATE_PROSPECTUS_ISSUANCE'
        if any(k in full_text for k in FIN_CAPITAL_STRUCTURE_KEYWORDS):
            return 'FIN_CAPITAL_STRUCTURE'

    # Fallback
    return 'CORP_MISC_UPDATE'


def process_documents(input_file: str) -> tuple[list, list]:
    """
    Reads JSON docs from input_file, classifies them, and returns two lists:
    - successes: docs with a valid category_handle
    - failures: docs with errors or fallback classifications
    Each doc is annotated with 'input_file' and 'cluster_id'.
    Cleans up any NaN or non-string form_type values.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return [], []

    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    successes = []
    failures = []

    for doc in documents:
        # Annotate origin
        doc['input_file'] = os.path.basename(input_file)
        doc['cluster_id'] = doc.get('cluster')

        # Clean up form_type: use only string values, else extract or None
        raw_ft = doc.get('form_type')
        form_type = raw_ft if isinstance(raw_ft, str) else extract_form_symbol(doc.get('text', ''))
        doc['form_type'] = form_type or None

        if not form_type:
            doc['classification_status'] = 'Error: Form symbol not found.'
            failures.append(doc)
            continue

        event_ids_str = normalize_event_ids(doc.get('eventIds', ''))
        category = classify_document(form_type, event_ids_str, doc.get('text', ''))

        if category == 'CORP_MISC_UPDATE':
            doc['classification_status'] = 'Fallback: CORP_MISC_UPDATE'
            failures.append(doc)
        else:
            doc['category_handle'] = category
            successes.append(doc)

    return successes, failures


def main():
    parser = argparse.ArgumentParser(description='Classify JSON documents into categories.')
    parser.add_argument('inputs', nargs='+', help='Paths to input JSON files')
    parser.add_argument('--success', default='classified.json', help='Output for successes')
    parser.add_argument('--failure', default='unclassified.json', help='Output for failures')
    args = parser.parse_args()

    all_successes, all_failures = [], []
    for inp in args.inputs:
        s, f = process_documents(inp)
        all_successes.extend(s)
        all_failures.extend(f)

    # Add separate running counts for successes and failures
    for idx, doc in enumerate(all_successes, start=1):
        doc['success_count'] = idx
    for idx, doc in enumerate(all_failures, start=1):
        doc['failure_count'] = idx

    # Write valid JSON (no NaN), preserving Hebrew
    with open(args.success, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(all_successes, f, ensure_ascii=False, indent=2)
    with open(args.failure, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(all_failures, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_successes)} successes to {args.success}")
    print(f"Wrote {len(all_failures)} failures to {args.failure}")

if __name__ == '__main__':
    main()
