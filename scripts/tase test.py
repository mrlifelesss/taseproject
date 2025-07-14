import http.client
import json
import pprint
import csv

conn = http.client.HTTPSConnection("datawise.tase.co.il")

headers = {
    'accept': "application/json",
    'accept-language': "en-US",
    'apikey': "[READCTED]"
    }

conn.request("GET", "/v1/basic-indices/indices-list", headers=headers)

res = conn.getresponse()
data = res.read()

decoded = data.decode("utf-8")
parsed = json.loads(decoded)

# Assuming 'data' is the response content
parsed = json.loads(data.decode("utf-8"))
indices = parsed['indicesList']['result']

# Define full file path
csv_path = r"C:\Users\ronbe\OneDrive\מסמכים\tase_project\indices_list.csv"

# Write to CSV at the specified location
with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=indices[0].keys())
    writer.writeheader()
    writer.writerows(indices)

print(f"CSV file created at: {csv_path}")
