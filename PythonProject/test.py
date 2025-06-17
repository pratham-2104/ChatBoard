import requests

def get_embedding(text, model="mistral"):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text}
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
embedding = get_embedding("Hello, how are you?")
print(embedding)
