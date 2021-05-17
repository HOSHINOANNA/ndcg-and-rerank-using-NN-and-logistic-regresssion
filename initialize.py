import requests

url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
r = requests.get(url)
with open("GoogleNews-vectors-negative300.bin.gz","wb") as f:
    f.write(r.content)