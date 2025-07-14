import boto3

client = boto3.client('dynamodb')
resp = client.describe_table(TableName='CompanyDisclosuresHebrew')
print("ItemCount from DescribeTable:", resp['Table']['ItemCount'])
print(resp["Table"]["KeySchema"])
