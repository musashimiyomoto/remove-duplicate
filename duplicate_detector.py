import pandas as pd
import numpy as np
from unidecode import unidecode
import re
from typing import List, Dict, Tuple, Optional
import colorsys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Детектор дубликатов на основе векторизации и косинусного расстояния
        
        Args:
            similarity_threshold: порог схожести от 0 до 1 (по умолчанию 0.85)
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None
        self.stemmer = SnowballStemmer('russian')
        
        # Инициализация NLTK данных
        self._init_nltk()
        
        # Регулярные выражения для нормализации текста
        self.common_words = {
            'org_forms': r'\b(ип|ооо|оао|зао|тов|ltd|llc|inc|corporation|corp|company|co)\b',
            'venue_types': r'\b(кафе|ресторан|бар|столовая|магазин|торговая точка|тт|cafe|restaurant|bar|shop|store)\b',
            'punctuation': r'[.,;:!?()"\'\-\№#@$%^&*+={}|\\`~<>/]',
            'extra_spaces': r'\s+',
            'postal_codes': r'^\d{5,6},?\s*',
            'numbers': r'\d+',
            'address_abbreviations': {
                r'\bг\b': 'город',
                r'\bул\b': 'улица', 
                r'\bпр\b': 'проспект',
                r'\bд\b': 'дом',
                r'\bстр\b': 'строение',
                r'\bк\b': 'корпус',
                r'\bобл\b': 'область',
                r'\bр-н\b': 'район',
                r'\bпос\b': 'поселок',
                r'\bс\b': 'село',
                r'\bдер\b': 'деревня',
                r'\bпл\b': 'площадь',
                r'\bпер\b': 'переулок',
                r'\bш\b': 'шоссе',
                r'\bнаб\b': 'набережная'
            }
        }
        
        # Русские стоп-слова
        self.russian_stopwords = {
            'и', 'в', 'на', 'с', 'по', 'за', 'от', 'до', 'для', 'при', 'о', 
            'об', 'под', 'над', 'между', 'через', 'без', 'из', 'к', 'у', 'т',
            'но', 'а', 'что', 'как', 'не', 'же', 'ли', 'бы', 'да', 'или'
        }
    
    def _init_nltk(self):
        """Инициализация NLTK данных"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

    def generate_colors(self, num_colors: int, is_dark_theme: bool = False) -> List[str]:
        """Генерация цветов для групп дубликатов"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            
            if is_dark_theme:
                saturation = 0.6 + (i % 3) * 0.1
                lightness = 0.4 + (i % 4) * 0.1
            else:
                saturation = 0.4 + (i % 3) * 0.1
                lightness = 0.85 + (i % 3) * 0.05
            
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
        return colors

    def preprocess_text(self, text: str, remove_org_forms: bool = True, remove_venue_types: bool = True) -> str:
        """
        Предобработка текста для векторизации
        
        Args:
            text: исходный текст
            remove_org_forms: убирать ли организационно-правовые формы
            remove_venue_types: убирать ли типы заведений
            
        Returns:
            обработанный текст
        """
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).strip().lower()
        
        # Удаляем пунктуацию
        text = re.sub(self.common_words['punctuation'], ' ', text)
        
        # Удаляем организационно-правовые формы
        if remove_org_forms:
            text = re.sub(self.common_words['org_forms'], '', text)
        
        # Удаляем типы заведений
        if remove_venue_types:
            text = re.sub(self.common_words['venue_types'], '', text)
        
        # Удаляем числа (номера домов, телефоны и т.д.)
        text = re.sub(self.common_words['numbers'], '', text)
        
        # Транслитерация
        try:
            text = unidecode(text)
        except:
            pass
        
        # Токенизация
        try:
            tokens = word_tokenize(text, language='russian')
        except:
            tokens = text.split()
        
        # Убираем стоп-слова и короткие слова
        tokens = [token for token in tokens 
                 if len(token) > 2 and token not in self.russian_stopwords]
        
        # Стемминг
        try:
            tokens = [self.stemmer.stem(token) for token in tokens]
        except:
            pass
        
        # Убираем лишние пробелы
        result = ' '.join(tokens)
        result = re.sub(self.common_words['extra_spaces'], ' ', result).strip()
        
        return result

    def normalize_address(self, address: str) -> str:
        """Нормализация адреса"""
        if pd.isna(address) or not address:
            return ""
        
        address = str(address).lower().strip()
        
        # Убираем почтовые коды
        address = re.sub(self.common_words['postal_codes'], '', address)
        
        # Заменяем сокращения
        for pattern, replacement in self.common_words['address_abbreviations'].items():
            address = re.sub(pattern, replacement, address)
        
        # Убираем пунктуацию и номера
        address = re.sub(r'[,.\-№]', ' ', address)
        address = re.sub(self.common_words['numbers'], '', address)
        
        # Убираем лишние пробелы
        address = re.sub(self.common_words['extra_spaces'], ' ', address).strip()
        
        return address

    def create_combined_features(self, df: pd.DataFrame, name_column: str, address_column: Optional[str] = None) -> List[str]:
        """
        Создание объединенных признаков для векторизации
        
        Args:
            df: DataFrame с данными
            name_column: колонка с названиями
            address_column: колонка с адресами (опционально)
            
        Returns:
            список обработанных текстов для векторизации
        """
        combined_features = []
        
        for idx, row in df.iterrows():
            # Обрабатываем название
            name_text = self.preprocess_text(row[name_column])
            
            # Обрабатываем адрес, если есть
            address_text = ""
            if address_column and address_column in df.columns:
                address_text = self.normalize_address(row[address_column])
            
            # Объединяем название и адрес с весами
            # Название важнее адреса, поэтому дублируем его
            combined_text = f"{name_text} {name_text} {address_text}".strip()
            combined_features.append(combined_text)
        
        return combined_features

    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Вычисление матрицы схожести с использованием TF-IDF и косинусного расстояния
        
        Args:
            texts: список текстов для сравнения
            
        Returns:
            матрица схожести
        """
        # Создаем TF-IDF векторизатор
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # используем униграммы и биграммы
            min_df=1,
            max_df=0.95,
            lowercase=True,
            analyzer='word'
        )
        
        # Векторизуем тексты
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Вычисляем косинусное сходство
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix

    def find_duplicates(self, df: pd.DataFrame, name_column: str, address_column: Optional[str] = None, id_column: Optional[str] = None) -> Tuple[List[List[int]], Dict]:
        """
        Поиск дубликатов с использованием векторизации
        
        Args:
            df: DataFrame с данными
            name_column: колонка с названиями
            address_column: колонка с адресами (опционально)
            id_column: колонка с ID (опционально, не используется)
            
        Returns:
            (группы дубликатов, статистика)
        """
        if df.empty:
            return [], {'total_records': 0, 'duplicate_groups': 0, 'duplicate_records': 0, 'unique_records': 0}
        
        # Создаем объединенные признаки
        combined_features = self.create_combined_features(df, name_column, address_column)
        
        # Фильтруем пустые тексты
        valid_indices = [i for i, text in enumerate(combined_features) if text.strip()]
        
        if not valid_indices:
            return [], {'total_records': len(df), 'duplicate_groups': 0, 'duplicate_records': 0, 'unique_records': len(df)}
        
        # Получаем только валидные тексты
        valid_texts = [combined_features[i] for i in valid_indices]
        
        # Вычисляем матрицу схожести
        similarity_matrix = self.calculate_similarity_matrix(valid_texts)
        
        # Находим группы дубликатов
        duplicate_groups = []
        processed_indices = set()
        
        for i, original_idx in enumerate(valid_indices):
            if original_idx in processed_indices:
                continue
                
            current_group = [original_idx]
            
            for j, compare_idx in enumerate(valid_indices[i+1:], i+1):
                if compare_idx in processed_indices:
                    continue
                
                # Проверяем схожесть
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.similarity_threshold:
                    current_group.append(compare_idx)
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                processed_indices.update(current_group)
        
        # Собираем статистику
        stats = {
            'total_records': len(df),
            'duplicate_groups': len(duplicate_groups),
            'duplicate_records': sum(len(group) for group in duplicate_groups),
            'unique_records': len(df) - sum(len(group) for group in duplicate_groups)
        }
        
        return duplicate_groups, stats

    def create_grouped_dataframe(self, df: pd.DataFrame, duplicate_groups: List[List[int]]) -> pd.DataFrame:
        """Создание DataFrame с группировкой дубликатов"""
        if not duplicate_groups:
            result_df = df.copy()
            result_df['Duplicate_Group'] = 0
            return result_df
        
        grouped_data = []
        group_counter = 1
        
        for group in duplicate_groups:
            for row_idx in group:
                row_data = df.iloc[row_idx].to_dict()
                row_data['Duplicate_Group'] = group_counter
                grouped_data.append(row_data)
            group_counter += 1
        
        # Добавляем уникальные записи
        remaining_indices = set(range(len(df))) - set(idx for group in duplicate_groups for idx in group)
        for row_idx in remaining_indices:
            row_data = df.iloc[row_idx].to_dict()
            row_data['Duplicate_Group'] = 0
            grouped_data.append(row_data)
        
        result_df = pd.DataFrame(grouped_data)
        result_df = result_df.sort_values(['Duplicate_Group', result_df.columns[0]], ascending=[False, True])
        
        return result_df

    def create_styled_dataframe(self, df: pd.DataFrame, duplicate_groups: List[List[int]], is_dark_theme: bool = False) -> Dict:
        """Создание стилизованного DataFrame для отображения"""
        grouped_df = self.create_grouped_dataframe(df, duplicate_groups)
        
        if not duplicate_groups:
            return {
                "data": grouped_df.values.tolist(),
                "headers": grouped_df.columns.tolist(),
                "metadata": {"styling": []},
            }
        
        colors = self.generate_colors(len(duplicate_groups), is_dark_theme)
        
        styling = []
        for _ in range(len(grouped_df)):
            styling.append([""] * len(grouped_df.columns))
        
        for idx, row in grouped_df.iterrows():
            group_num = row['Duplicate_Group']
            if group_num > 0:
                color = colors[(group_num - 1) % len(colors)]
                for col_idx in range(len(grouped_df.columns)):
                    styling[idx][col_idx] = f"background-color: {color};"
        
        return {
            "data": grouped_df.values.tolist(),
            "headers": grouped_df.columns.tolist(),
            "metadata": {
                "styling": styling,
            },
        }

    def find_name_column(self, df: pd.DataFrame) -> Optional[str]:
        """Автоматически находит колонку с названиями/наименованиями"""
        columns = [col.lower() for col in df.columns]
        
        # Приоритетные ключевые слова для поиска колонки с названиями
        name_keywords = [
            'назван', 'наимен', 'имя', 'name', 'title', 'компан', 'организац', 
            'предприят', 'фирм', 'company', 'organization', 'firm'
        ]
        
        for i, col in enumerate(columns):
            for keyword in name_keywords:
                if keyword in col:
                    return df.columns[i]
        
        return None
    
    def find_address_column(self, df: pd.DataFrame) -> Optional[str]:
        """Автоматически находит колонку с адресами"""
        columns = [col.lower() for col in df.columns]
        
        # Ключевые слова для поиска колонки с адресами
        address_keywords = ['адрес', 'address', 'местоположен', 'location']
        
        for i, col in enumerate(columns):
            for keyword in address_keywords:
                if keyword in col:
                    return df.columns[i]
        
        return None

    def get_similarity_info(self) -> str:
        """Получение информации о методе определения схожести"""
        return f"""
        🔬 **Метод определения дубликатов:**
        - Векторизация с TF-IDF
        - Косинусное расстояние
        - Порог схожести: {self.similarity_threshold:.2f}
        - Предобработка: стемминг, удаление стоп-слов
        """ 