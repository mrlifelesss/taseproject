import os
import boto3
from boto3.dynamodb.conditions import Attr

# configuration
TABLE_NAME = os.environ.get('DDB_TABLE', 'CompanyDisclosures')

dynamodb = boto3.resource('dynamodb')
table     = dynamodb.Table(TABLE_NAME)

def fix_ca_to_he():
    # 1) scan for all items where language == 'ca'
    paginator = table.meta.client.get_paginator('scan')
    scan_kwargs = {
        'TableName': TABLE_NAME,
        'FilterExpression': Attr('language').eq('ca'),
        'ProjectionExpression': 'reportId, publicationDate'
    }

    for page in paginator.paginate(**scan_kwargs):
        for item in page['Items']:
            pk = item['reportId']
            sk = item['publicationDate']
            print(f"Updating {pk} @ {sk} → language='he'")
            # 2) update each one
            table.update_item(
                Key={ 'reportId': pk, 'publicationDate': sk },
                UpdateExpression='SET #lang = :h',
                ExpressionAttributeNames={ '#lang': 'language' },
                ExpressionAttributeValues={ ':h': 'he' }
            )

if __name__ == '__main__':
    fix_ca_to_he()
    print("Done.")
