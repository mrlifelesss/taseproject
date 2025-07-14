import os
import boto3
from botocore.exceptions import ClientError

AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
TABLE_NAME   = "CompanyDisclosuresHebrew"

ddb   = boto3.resource("dynamodb", region_name=AWS_REGION)
table = ddb.Table(TABLE_NAME)

def scan_all_items():
    last_key = None
    while True:
        scan_kwargs = {}
        if last_key:
            scan_kwargs["ExclusiveStartKey"] = last_key

        try:
            resp = table.scan(**scan_kwargs)
        except ClientError as e:
            print("Scan error:", e)
            return

        for item in resp.get("Items", []):
            yield item

        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break

def find_matching_reports(target_event_name):
    matches = []
    for item in scan_all_items():
        issuer = item.get("issuerName")
        date   = item.get("publicationDate")

        # The raw structure (as you saw) is:
        #  "events": { "M": { "event": { "L": [ { "M": { "eventId": {"N":"..."}, "eventName": {"S":"..."} } }, ... ] } } }
        low_events = item.get("events")
        if not isinstance(low_events, dict):
            continue
        top_m = low_events.get("M")
        if not isinstance(top_m, dict):
            continue

        list_wrapper = top_m.get("event", {})
        if not isinstance(list_wrapper, dict):
            continue

        array_of_wrappers = list_wrapper.get("L", [])
        if not isinstance(array_of_wrappers, list):
            continue

        # Now iterate every element, which should be { "M": { "eventId":{…}, "eventName":{…} } }
        for elem in array_of_wrappers:
            m = elem.get("M")
            if not isinstance(m, dict):
                continue

            en = m.get("eventName", {}).get("S")
            if en == target_event_name:
                # We found a match
                matches.append({"issuerName": issuer, "publicationDate": date})
                break  # stop checking further events in this item

    return matches

if __name__ == "__main__":
    target = "אירועים ועסקאות"
    found = find_matching_reports(target)
    print(f"Reports whose events.event[].eventName = '{target}':")
    for rec in found:
        print(" •", rec["issuerName"], "@", rec["publicationDate"])
    print(f"\nTotal matches: {len(found)}")
