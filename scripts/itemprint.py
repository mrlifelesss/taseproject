import boto3
import json
from decimal import Decimal

def decimal_default(o):
    """
    Called by json.dumps for any object it cannot serialize by default.
    If it’s a Decimal, convert to int (if possible) or float.
    Otherwise, raise TypeError.
    """
    if isinstance(o, Decimal):
        if o == o.to_integral_value():
            return int(o)
        else:
            return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def main():
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('CompanyDisclosuresHebrew')

    # Fetch one example item
    resp = table.scan(Limit=1)
    items = resp.get('Items', [])
    if not items:
        print("No items found.")
        return

    item = items[0]

    # Pass decimal_default as the default= argument
    with open('sample_item.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False, indent=2, default=decimal_default))

if __name__ == '__main__':
    main()