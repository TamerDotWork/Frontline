import requests

URL = "http://127.0.0.1:8000/chat"

question = "What is Love???"

payload = {
    "model": "gemma2:9b-instruct-q5_0",
    "stream": False,
    "messages": [
        {"role": "user", "content": question}
    ]
}

response = requests.post(URL, json=payload)
data = response.json()

print("Response:", data)
