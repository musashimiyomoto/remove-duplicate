import nltk
import sys


def setup_nltk():
    try:
        print("🔄 Loading NLTK data for enhanced text processing...")

        print("- Loading punkt (tokenization)...")
        nltk.download("punkt", quiet=True)

        print("- Loading stopwords (Russian & English)...")
        nltk.download("stopwords", quiet=True)
        
        print("- Loading punkt_tab (enhanced tokenization)...")
        nltk.download("punkt_tab", quiet=True)

        try:
            from nltk.corpus import stopwords
            from nltk.stem import SnowballStemmer
            
            russian_stopwords = stopwords.words('russian')
            SnowballStemmer('russian')
            print(f"- Russian stopwords loaded: {len(russian_stopwords)} words")
            print("- Russian stemmer ready")
            
        except Exception as e:
            print(f"⚠️ Russian language resources issue: {e}")

        print("✅ NLTK setup completed successfully!")
        print("📈 This will improve duplicate detection quality by:")
        print("   • Better text tokenization")
        print("   • Removing Russian stop words")  
        print("   • Word stemming for root forms")
        return True

    except Exception as e:
        print(f"❌ Error loading NLTK data: {e}")
        return False


if __name__ == "__main__":
    success = setup_nltk()
    sys.exit(0 if success else 1)
