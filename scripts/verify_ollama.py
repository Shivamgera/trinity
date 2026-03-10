"""Verify ollama is running and Llama model responds."""

import requests


def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": "In one sentence, what is a stock market?",
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Response: {result.get('response', 'NO RESPONSE')[:200]}")
    print("Ollama verification: PASSED")


if __name__ == "__main__":
    test_ollama()
