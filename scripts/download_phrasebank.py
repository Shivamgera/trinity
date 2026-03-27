"""Download FinancialPhraseBank dataset for Analyst validation.

Downloads the FinancialPhraseBank (Malo et al. 2014) from Hugging Face Hub,
which contains ~4,845 financial sentences with sentiment labels.
Used to validate the Analyst LLM's sentiment understanding.

Downloads the zip archive and extracts the "Sentences_AllAgree.txt" subset.
"""

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import requests


PHRASEBANK_URL = (
    "https://huggingface.co/datasets/financial_phrasebank/resolve/main/"
    "data/FinancialPhraseBank-v1.0.zip"
)
TARGET_FILE = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"


def _parse_phrasebank_line(line: str) -> tuple[str, str] | None:
    """Parse a single line from the raw FinancialPhraseBank file.

    Format: '<sentence>@<label>' where label ∈ {positive, negative, neutral}.
    """
    for label in ("positive", "negative", "neutral"):
        suffix = f"@{label}"
        if line.rstrip().endswith(suffix):
            sentence = line.rstrip()[: -len(suffix)].strip()
            return sentence, label
    return None


def main():
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "raw" / "financial_phrasebank.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading FinancialPhraseBank from Hugging Face Hub...")
    resp = requests.get(PHRASEBANK_URL, timeout=60)
    resp.raise_for_status()

    # Extract the target file from the zip
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        print(f"Zip contents: {zf.namelist()}")
        raw_bytes = zf.read(TARGET_FILE)

    # Parse the raw text file (Latin-1 encoded)
    text = raw_bytes.decode("latin-1")
    lines = text.strip().split("\n")

    rows: list[tuple[str, str]] = []
    for line in lines:
        parsed = _parse_phrasebank_line(line)
        if parsed:
            rows.append(parsed)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label"])
        for sentence, label in rows:
            writer.writerow([sentence, label])

    print(f"Saved {len(rows)} sentences to {output_path}")

    # Print label distribution
    labels = [label for _, label in rows]
    counts = Counter(labels)
    print(f"Label distribution: {dict(counts)}")


if __name__ == "__main__":
    main()
