
import requests
 
URL = ("http://localhost:8000/chat")

question = "What is Sex?"

payload = {
    "contents": [
        {"parts": [{"text": question}]}
    ]
}

response = requests.post(URL, json=payload)
data = response.json()

