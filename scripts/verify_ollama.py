"""Verify ollama is running and Llama model responds."""

import sys
import requests


OLLAMA_BASE = "http://localhost:11434"


def _available_models():
    """Return list of model names pulled in Ollama."""
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def _pick_model(preferred="llama3.2:latest"):
    """Return *preferred* if available, else first llama model, else first model."""
    models = _available_models()
    if not models:
        return None
    if preferred in models:
        return preferred
    # fall back to any llama variant
    llama = [m for m in models if "llama" in m.lower()]
    return llama[0] if llama else models[0]


def test_ollama():
    # 1. Check Ollama is reachable
    try:
        requests.get(OLLAMA_BASE, timeout=5)
    except requests.ConnectionError:
        print("ERROR: Ollama is not running. Start it with `ollama serve`.")
        sys.exit(1)

    # 2. Pick a model
    model = _pick_model()
    if model is None:
        print("ERROR: No models found. Pull one with `ollama pull llama3.1:8b`.")
        sys.exit(1)

    print(f"Using model: {model}")

    # 3. Generate
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": "In one sentence, what is a stock market?",
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    print(f"Response: {result.get('response', 'NO RESPONSE')[:200]}")
    print("Ollama verification: PASSED")


if __name__ == "__main__":
    test_ollama()
