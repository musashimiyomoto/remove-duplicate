import pandas as pd
import numpy as np
from unidecode import unidecode
import re
from typing import List, Dict, Tuple, Optional
import colorsys
from difflib import SequenceMatcher
import warnings

warnings.filterwarnings("ignore")


class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.80):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None

        # Регулярные выражения для нормализации
        self.patterns = {
            "extra_spaces": r"\s+",
            "punctuation": r"[^\w\s\d]",
            "org_forms": r"\b(ооо|оао|зао|ип|тов|ltd|llc|inc|corp|co|общество|предприятие)\b",
            "address_numbers": r"\b\d+[\/\-]?\d*\b",  # номера домов, квартир
            "postal_codes": r"\b\d{5,6}\b",  # почтовые индексы
        }

    def generate_colors(
        self, num_colors: int, is_dark_theme: bool = False
    ) -> List[str]:
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
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors

    def normalize_text(self, text: str, preserve_numbers: bool = True) -> str:
        """
        Улучшенная нормализация текста

        Args:
            text: исходный текст
            preserve_numbers: сохранять ли числа (важно для адресов)
        """
        if pd.isna(text) or not text:
            return ""

        text = str(text).strip().lower()

        # Убираем только некритичную пунктуацию, сохраняя числа и важные символы
        if preserve_numbers:
            text = re.sub(r'[,.\-№"\'()]', " ", text)
        else:
            text = re.sub(self.patterns["punctuation"], " ", text)

        # Убираем лишние пробелы
        text = re.sub(self.patterns["extra_spaces"], " ", text).strip()

        return text

    def normalize_name(self, name: str) -> str:
        """Специальная нормализация для названий организаций"""
        if pd.isna(name) or not name:
            return ""

        name = str(name).strip().lower()

        # Убираем организационные формы
        name = re.sub(self.patterns["org_forms"], "", name, flags=re.IGNORECASE)

        # Убираем пунктуацию
        name = re.sub(r"[^\w\s]", " ", name)
        name = re.sub(self.patterns["extra_spaces"], " ", name).strip()

        return name

    def extract_address_numbers(self, address: str) -> List[str]:
        """Извлекает номера домов из адреса"""
        if pd.isna(address) or not address:
            return []

        # Находим все числа в адресе (номера домов, квартир и т.д.)
        numbers = re.findall(self.patterns["address_numbers"], str(address))
        return numbers

    def string_similarity(self, str1: str, str2: str) -> float:
        """Вычисление схожести строк через SequenceMatcher"""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1, str2).ratio()

    def name_similarity(self, name1: str, name2: str) -> float:
        """Специальная проверка схожести названий"""
        if not name1 or not name2:
            return 0.0

        # Нормализуем названия
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)

        # Точное совпадение после нормализации
        if norm1 == norm2 and len(norm1) > 2:
            return 1.0

        # Базовая схожесть
        basic_sim = self.string_similarity(norm1, norm2)

        # Проверяем, содержит ли одно название другое
        if norm1 in norm2 or norm2 in norm1:
            return max(basic_sim, 0.9)

        return basic_sim

    def address_similarity(self, addr1: str, addr2: str) -> float:
        """Проверка схожести адресов с учетом номеров домов"""
        if not addr1 or not addr2:
            return 0.0

        # Нормализуем адреса
        norm1 = self.normalize_text(addr1, preserve_numbers=True)
        norm2 = self.normalize_text(addr2, preserve_numbers=True)

        # Извлекаем номера домов
        numbers1 = self.extract_address_numbers(addr1)
        numbers2 = self.extract_address_numbers(addr2)

        # Если у обоих адресов есть номера домов и они разные, то адреса точно разные
        if numbers1 and numbers2:
            common_numbers = set(numbers1).intersection(set(numbers2))
            if not common_numbers:
                return 0.0  # Разные номера домов = разные адреса

        # Базовая схожесть адресов
        basic_sim = self.string_similarity(norm1, norm2)

        # Если один адрес содержит номер дома, а другой нет, то снижаем схожесть
        if (numbers1 and not numbers2) or (numbers2 and not numbers1):
            basic_sim *= 0.8  # Снижаем на 20%

        return basic_sim

    def are_duplicates(
        self,
        row1: pd.Series,
        row2: pd.Series,
        name_col: str,
        address_col: Optional[str] = None,
    ) -> bool:
        """
        Основная логика определения дубликатов

        Args:
            row1, row2: строки для сравнения
            name_col: колонка с названиями
            address_col: колонка с адресами

        Returns:
            True если записи являются дубликатами
        """
        name1 = str(row1[name_col]) if pd.notna(row1[name_col]) else ""
        name2 = str(row2[name_col]) if pd.notna(row2[name_col]) else ""

        # Проверяем схожесть названий
        name_sim = self.name_similarity(name1, name2)

        # Если названия очень разные, то это точно не дубликаты
        if name_sim < 0.6:
            return False

        # Если есть адреса, проверяем их
        if address_col and address_col in row1.index and address_col in row2.index:
            addr1 = str(row1[address_col]) if pd.notna(row1[address_col]) else ""
            addr2 = str(row2[address_col]) if pd.notna(row2[address_col]) else ""

            addr_sim = self.address_similarity(addr1, addr2)

            # Если адреса очень разные (например, разные номера домов), то НЕ дубликаты
            if addr_sim == 0.0:
                return False

            # Комбинированная оценка с более мягкими требованиями
            combined_sim = name_sim * 0.7 + addr_sim * 0.3

            # Условия для дубликатов:
            # 1. Очень высокая схожесть названий (>0.85) ИЛИ
            # 2. Хорошая схожесть названий (>0.7) И хорошая схожесть адресов (>0.7) ИЛИ
            # 3. Комбинированная оценка >0.75
            return (
                name_sim >= 0.85
                or (name_sim >= 0.7 and addr_sim >= 0.7)
                or combined_sim >= 0.75
            )
        else:
            # Если нет адресов, то названия должны быть очень похожими
            return name_sim >= 0.85

    def find_duplicates(
        self,
        df: pd.DataFrame,
        name_column: str,
        address_column: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> Tuple[List[List[int]], Dict]:
        """
        Поиск дубликатов с улучшенной логикой

        Args:
            df: DataFrame с данными
            name_column: колонка с названиями
            address_column: колонка с адресами (опционально)
            id_column: колонка с ID (опционально, не используется)

        Returns:
            (группы дубликатов, статистика)
        """
        if df.empty:
            return [], {
                "total_records": 0,
                "duplicate_groups": 0,
                "duplicate_records": 0,
                "unique_records": 0,
            }

        # Находим группы дубликатов
        duplicate_groups = []
        processed_indices = set()

        for i in range(len(df)):
            if i in processed_indices:
                continue

            current_group = [i]

            for j in range(i + 1, len(df)):
                if j in processed_indices:
                    continue

                # Проверяем, являются ли записи дубликатами
                if self.are_duplicates(
                    df.iloc[i], df.iloc[j], name_column, address_column
                ):
                    current_group.append(j)

            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                processed_indices.update(current_group)

        # Собираем статистику
        stats = {
            "total_records": len(df),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_records": sum(len(group) for group in duplicate_groups),
            "unique_records": len(df) - sum(len(group) for group in duplicate_groups),
        }

        return duplicate_groups, stats

    def create_grouped_dataframe(
        self, df: pd.DataFrame, duplicate_groups: List[List[int]]
    ) -> pd.DataFrame:
        """Создание DataFrame с группировкой дубликатов"""
        if not duplicate_groups:
            result_df = df.copy()
            result_df["Duplicate_Group"] = 0
            return result_df

        grouped_data = []
        group_counter = 1

        for group in duplicate_groups:
            for row_idx in group:
                row_data = df.iloc[row_idx].to_dict()
                row_data["Duplicate_Group"] = group_counter
                grouped_data.append(row_data)
            group_counter += 1

        # Добавляем уникальные записи
        remaining_indices = set(range(len(df))) - set(
            idx for group in duplicate_groups for idx in group
        )
        for row_idx in remaining_indices:
            row_data = df.iloc[row_idx].to_dict()
            row_data["Duplicate_Group"] = 0
            grouped_data.append(row_data)

        result_df = pd.DataFrame(grouped_data)
        result_df = result_df.sort_values(
            ["Duplicate_Group", result_df.columns[0]], ascending=[False, True]
        )

        return result_df

    def create_styled_dataframe(
        self,
        df: pd.DataFrame,
        duplicate_groups: List[List[int]],
        is_dark_theme: bool = False,
    ) -> Dict:
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
            group_num = row["Duplicate_Group"]
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
            "назван",
            "наимен",
            "имя",
            "name",
            "title",
            "компан",
            "организац",
            "предприят",
            "фирм",
            "company",
            "organization",
            "firm",
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
        address_keywords = ["адрес", "address", "местоположен", "location"]

        for i, col in enumerate(columns):
            for keyword in address_keywords:
                if keyword in col:
                    return df.columns[i]

        return None

    def get_similarity_info(self) -> str:
        """Получение информации о методе определения схожести"""
        return f"""
        🔬 **Метод определения дубликатов:**
        - Улучшенный алгоритм с проверкой адресов
        - Учет номеров домов и различий в адресах
        - Строгие требования к схожести названий (>80%)
        - Комбинированная оценка: название + адрес
        """
