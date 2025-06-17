import pandas as pd
import re
from thefuzz import fuzz
import networkx as nx
from collections import Counter
import colorsys
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.80):
        self.similarity_threshold = similarity_threshold

    def generate_colors(
        self, num_colors: int, is_dark_theme: bool = False
    ) -> List[str]:
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
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors

    def normalize_text(self, text, sort_words=True):
        """Нормализация текста с сортировкой слов по умолчанию"""
        if not isinstance(text, str): 
            return ""
        text = text.lower()
        replacements = {
            r'\bобщество с ограниченной ответственностью\b': 'ооо', 
            r'(\bип\b)|(\bиндивидуальный предприниматель\b)': ' ',
            r'\bгород\b': 'г', 
            r'\bулица\b': 'ул', 
            r'\bдом\b': 'д', 
            r'\bстроение\b': 'стр', 
            r'\bкорпус\b': 'к',
        }
        for p, r in replacements.items(): 
            text = re.sub(p, r, text)
        text = re.sub(r'[^a-zа-я0-9\s]', ' ', text)
        words = text.split()
        if sort_words: 
            words = sorted(words)
        return ' '.join(words).strip()

    def normalize_brand_name(self, text):
        """Нормализация брендового имени без учета типовых слов"""
        base_normalized = self.normalize_text(text, sort_words=False)
        generic_words = {'ооо', 'кафе', 'бар', 'ресторан', 'фастфуд', 'магазин', 'пиццерия', 'клуб', 'кофейня'}
        words = [word for word in base_normalized.split() if word not in generic_words]
        return ' '.join(sorted(words))

    # ==========================================================================
    # ОПРЕДЕЛЕНИЕ ЧЕТЫРЕХ НЕЗАВИСИМЫХ СУДЕЙ
    # ==========================================================================

    def get_duplicates_judge1_strict(self, df):
        """Судья 1: 'Строгий Контроль'."""
        duplicates = set()
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                name_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_name'], df.loc[id2, 'norm_name'])
                addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                if addr_sim > 95 and name_sim > 90:
                    duplicates.add(frozenset([id1, id2]))
        return duplicates

    def get_duplicates_judge2_geo(self, df):
        """Судья 2: 'Гео-Аналитик'."""
        duplicates = set()
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                if addr_sim > 97:
                    name_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_name'], df.loc[id2, 'norm_name'])
                    if name_sim > 70:
                        duplicates.add(frozenset([id1, id2]))
        return duplicates

    def get_duplicates_judge3_brand(self, df):
        """Судья 3: 'Анализ по Бренду'."""
        duplicates = set()
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                brand_name_sim = fuzz.token_set_ratio(df.loc[id1, 'brand_name'], df.loc[id2, 'brand_name'])
                if brand_name_sim > 95:
                    addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                    if addr_sim > 75:
                        duplicates.add(frozenset([id1, id2]))
        return duplicates

    def get_duplicates_judge4_weighted(self, df):
        """Судья 4: 'Интегратор' (Взвешенная оценка)."""
        duplicates = set()
        SIMILARITY_THRESHOLD = 88  # Порог для большей точности
        NAME_WEIGHT = 0.45
        ADDRESS_WEIGHT = 0.55
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                name_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_name'], df.loc[id2, 'norm_name'])
                addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                weighted_similarity = (name_sim * NAME_WEIGHT) + (addr_sim * ADDRESS_WEIGHT)
                if weighted_similarity >= SIMILARITY_THRESHOLD:
                    duplicates.add(frozenset([id1, id2]))
        return duplicates

    def find_duplicates(
        self,
        df: pd.DataFrame,
        name_column: str,
        address_column: Optional[str] = None,
        id_column: Optional[str] = None,
        min_votes: int = 2
    ) -> Tuple[List[List[int]], Dict]:
        """Основная функция поиска дубликатов с использованием консилиума из 4 судей"""
        
        if df.empty:
            return [], {
                "total_records": 0,
                "duplicate_groups": 0,
                "duplicate_records": 0,
                "unique_records": 0,
            }

        # Подготовка данных
        df_work = df.copy()
        
        # Если есть колонка ID, используем её, иначе создаём свою
        if id_column and id_column in df_work.columns:
            df_work['Id'] = df_work[id_column]
        else:
            df_work['Id'] = range(len(df_work))
        
        df_work = df_work.set_index('Id', drop=False)
        
        # Нормализация данных
        df_work['norm_name'] = df_work[name_column].apply(lambda x: self.normalize_text(str(x) if pd.notna(x) else "", True))
        if address_column and address_column in df_work.columns:
            df_work['norm_addr'] = df_work[address_column].apply(lambda x: self.normalize_text(str(x) if pd.notna(x) else "", True))
        else:
            df_work['norm_addr'] = ""
        df_work['brand_name'] = df_work[name_column].apply(lambda x: self.normalize_brand_name(str(x) if pd.notna(x) else ""))
        
        # Запуск всех судей
        results1 = self.get_duplicates_judge1_strict(df_work)
        results2 = self.get_duplicates_judge2_geo(df_work)
        results3 = self.get_duplicates_judge3_brand(df_work)
        results4 = self.get_duplicates_judge4_weighted(df_work)

        # Подсчёт голосов
        all_votes = Counter(results1)
        all_votes.update(results2)
        all_votes.update(results3)
        all_votes.update(results4)

        # Найдём пары, набравшие достаточное количество голосов
        final_duplicate_pairs = [list(pair) for pair, count in all_votes.items() if count >= min_votes]

        # Группировка с использованием NetworkX
        G = nx.Graph()
        G.add_edges_from(final_duplicate_pairs)
        connected_components = list(nx.connected_components(G))
        final_groups = [list(group) for group in connected_components if len(group) > 1]
        
        # Преобразуем ID обратно в индексы DataFrame
        final_groups_indices = []
        for group in final_groups:
            group_indices = []
            for id_val in group:
                # Найдём соответствующий индекс в исходном DataFrame
                matching_rows = df[df.index == id_val].index.tolist()
                if matching_rows:
                    group_indices.append(matching_rows[0])
                else:
                    # Если не найдено по ID, попробуем найти по позиции
                    if isinstance(id_val, int) and id_val < len(df):
                        group_indices.append(id_val)
            if group_indices:
                final_groups_indices.append(group_indices)

        # Статистика
        stats = {
            "total_records": len(df),
            "duplicate_groups": len(final_groups_indices),
            "duplicate_records": sum(len(group) for group in final_groups_indices),
            "unique_records": len(df) - sum(len(group) for group in final_groups_indices),
        }

        return final_groups_indices, stats

    def create_grouped_dataframe(self, df: pd.DataFrame, duplicate_groups: List[List[int]]) -> pd.DataFrame:
        """Создание DataFrame с группировкой дубликатов"""
        grouped_df = df.copy()
        grouped_df['Duplicate_Group'] = 'Unique'
        
        for i, group in enumerate(duplicate_groups):
            for idx in group:
                if idx < len(grouped_df):
                    grouped_df.loc[grouped_df.index[idx], 'Duplicate_Group'] = f'Group_{i+1}'
        
        # Сортировка: сначала группы дубликатов, потом уникальные
        grouped_df['sort_key'] = grouped_df['Duplicate_Group'].apply(
            lambda x: (0, x) if x != 'Unique' else (1, x)
        )
        grouped_df = grouped_df.sort_values('sort_key').drop('sort_key', axis=1)
        
        return grouped_df

    def create_styled_dataframe(self, df: pd.DataFrame, duplicate_groups: List[List[int]], is_dark_theme: bool = False) -> Dict:
        """Создание стилизованного DataFrame для отображения в интерфейсе"""
        if not duplicate_groups:
            return {
                "data": df.values.tolist(),
                "headers": df.columns.tolist(),
            }
        
        colors = self.generate_colors(len(duplicate_groups), is_dark_theme)
        
        # Создаём карту цветов для каждой строки
        row_colors = [''] * len(df)
        for i, group in enumerate(duplicate_groups):
            for idx in group:
                if idx < len(df):
                    row_colors[idx] = colors[i % len(colors)]
        
        # Подготавливаем данные с цветовой разметкой
        styled_data = []
        for i, row in enumerate(df.values.tolist()):
            if row_colors[i]:
                styled_row = [f"<div style='background-color: {row_colors[i]}; padding: 5px;'>{cell}</div>" for cell in row]
                styled_data.append(styled_row)
            else:
                styled_data.append(row)
        
        return {
            "data": styled_data,
            "headers": df.columns.tolist(),
        }

    def find_name_column(self, df: pd.DataFrame) -> Optional[str]:
        """Поиск колонки с названиями"""
        name_keywords = ['name', 'название', 'наименование', 'title', 'company', 'компания', 'тт']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in name_keywords):
                return col
        
        # Если не найдено, возвращаем первую колонку
        return df.columns[0] if len(df.columns) > 0 else None

    def find_address_column(self, df: pd.DataFrame) -> Optional[str]:
        """Поиск колонки с адресами"""
        address_keywords = ['address', 'адрес', 'location', 'место', 'расположение']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in address_keywords):
                return col
        
        return None
