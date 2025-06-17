#!/usr/bin/env python3

import nltk
import sys


def setup_nltk():
    try:
        print("Loading NLTK data...")

        print("- Loading punkt (tokenization)...")
        nltk.download("punkt", quiet=True)

        print("- Loading stopwords...")
        nltk.download("stopwords", quiet=True)

        print("✅ NLTK data loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading NLTK data: {e}")
        return False


if __name__ == "__main__":
    success = setup_nltk()
    sys.exit(0 if success else 1)
