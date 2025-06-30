import requests

url = "https://www.broadcastify.com/archives/feed/14744"

response = requests.get(url)

print(response.text)

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(response.text)