import pandas as pd
import numpy as np
from unidecode import unidecode
import re
from typing import List, Dict, Tuple, Optional
import colorsys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.70):
        """
        Детектор дубликатов с комбинацией нескольких методов
        
        Args:
            similarity_threshold: порог схожести от 0 до 1 (по умолчанию 0.70)
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None
        
        # Регулярные выражения для базовой нормализации
        self.patterns = {
            'extra_spaces': r'\s+',
            'punctuation': r'[^\w\s]',
            'numbers_only': r'^\d+$',
            'org_forms': r'\b(ооо|оао|зао|ип|тов|ltd|llc|inc|corp|co)\b',
        }

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

    def normalize_text(self, text: str) -> str:
        """Базовая нормализация текста"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).strip().lower()
        
        # Убираем лишние пробелы и пунктуацию
        text = re.sub(self.patterns['punctuation'], ' ', text)
        text = re.sub(self.patterns['extra_spaces'], ' ', text).strip()
        
        return text

    def normalize_text_strict(self, text: str) -> str:
        """Строгая нормализация текста для более точного сравнения"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).strip().lower()
        
        # Убираем организационные формы
        text = re.sub(self.patterns['org_forms'], '', text, flags=re.IGNORECASE)
        
        # Убираем пунктуацию и лишние пробелы
        text = re.sub(self.patterns['punctuation'], ' ', text)
        text = re.sub(self.patterns['extra_spaces'], ' ', text).strip()
        
        # Транслитерация для кириллицы
        try:
            text = unidecode(text)
        except:
            pass
        
        return text

    def string_similarity(self, str1: str, str2: str) -> float:
        """Вычисление схожести строк через SequenceMatcher"""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1, str2).ratio()

    def jaccard_similarity(self, str1: str, str2: str) -> float:
        """Вычисление схожести Жаккара на основе слов"""
        if not str1 or not str2:
            return 0.0
        
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)

    def calculate_combined_similarity(self, text1: str, text2: str) -> float:
        """Комбинированное вычисление схожести"""
        if not text1 or not text2:
            return 0.0
        
        # Базовая нормализация
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Строгая нормализация
        strict1 = self.normalize_text_strict(text1)
        strict2 = self.normalize_text_strict(text2)
        
        # Различные метрики схожести
        basic_sim = self.string_similarity(norm1, norm2)
        strict_sim = self.string_similarity(strict1, strict2)
        jaccard_sim = self.jaccard_similarity(norm1, norm2)
        
        # Проверка на точное совпадение после нормализации
        if strict1 == strict2 and len(strict1) > 2:
            return 1.0
        
        # Комбинированная оценка с весами
        combined_score = (basic_sim * 0.4 + strict_sim * 0.4 + jaccard_sim * 0.2)
        
        return combined_score

    def find_duplicates(self, df: pd.DataFrame, name_column: str, address_column: Optional[str] = None, id_column: Optional[str] = None) -> Tuple[List[List[int]], Dict]:
        """
        Поиск дубликатов с использованием комбинированного алгоритма
        
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
        
        # Создаем комбинированные тексты для сравнения
        combined_texts = []
        for idx, row in df.iterrows():
            name_text = str(row[name_column]) if pd.notna(row[name_column]) else ""
            address_text = ""
            
            if address_column and address_column in df.columns and pd.notna(row[address_column]):
                address_text = str(row[address_column])
            
            # Объединяем название и адрес
            combined_text = f"{name_text} {address_text}".strip()
            combined_texts.append(combined_text)
        
        # Находим группы дубликатов
        duplicate_groups = []
        processed_indices = set()
        
        for i in range(len(combined_texts)):
            if i in processed_indices:
                continue
            
            current_group = [i]
            
            for j in range(i + 1, len(combined_texts)):
                if j in processed_indices:
                    continue
                
                # Вычисляем схожесть
                similarity = self.calculate_combined_similarity(combined_texts[i], combined_texts[j])
                
                # Дополнительная проверка для названий
                name_similarity = self.calculate_combined_similarity(
                    str(df.iloc[i][name_column]) if pd.notna(df.iloc[i][name_column]) else "",
                    str(df.iloc[j][name_column]) if pd.notna(df.iloc[j][name_column]) else ""
                )
                
                # Считаем дубликатами если:
                # 1. Общая схожесть выше порога ИЛИ
                # 2. Схожесть названий очень высокая (>0.8)
                if similarity >= self.similarity_threshold or name_similarity >= 0.8:
                    current_group.append(j)
            
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
        - Комбинированный алгоритм
        - Схожесть строк + Жаккар + нормализация
        - Порог схожести: {self.similarity_threshold:.2f}
        - Дополнительная проверка названий (>0.8)
        """ 