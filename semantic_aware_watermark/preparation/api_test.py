import http.client
import json

# If you get the OpenAI API key, just use it!

sentence = 'Hello World.'
conn = http.client.HTTPSConnection("oa.api2d.net")
payload = json.dumps({ "model": "text-embedding-ada-002", "input": sentence })
headers = { 'Authorization': 'Your Key', 'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', 'Content-Type': 'application/json' }
conn.request("POST", "/v1/embeddings", payload, headers)
res = conn.getresponse()

res_data = res.read().decode("utf-8")
json_data = json.loads(res_data)

embedding = json_data['data'][0]['embedding']
print(embedding)
