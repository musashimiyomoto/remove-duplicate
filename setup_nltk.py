#!/usr/bin/env python3
"""
Скрипт для инициализации NLTK данных
Запустите его один раз после установки зависимостей
"""

import nltk
import sys

def setup_nltk():
    """Загружает необходимые NLTK данные"""
    try:
        print("Загрузка NLTK данных...")
        
        # Загружаем пакеты для токенизации
        print("- Загружаем punkt (токенизация)...")
        nltk.download('punkt', quiet=True)
        
        # Загружаем стоп-слова
        print("- Загружаем stopwords (стоп-слова)...")
        nltk.download('stopwords', quiet=True)
        
        print("✅ NLTK данные успешно загружены!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки NLTK данных: {e}")
        return False

if __name__ == "__main__":
    success = setup_nltk()
    sys.exit(0 if success else 1) 