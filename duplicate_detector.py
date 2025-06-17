import pandas as pd
import re
from thefuzz import fuzz
import networkx as nx
from collections import Counter
import colorsys
from typing import List, Dict, Tuple, Optional
import warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

warnings.filterwarnings("ignore")


class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.80):
        self.similarity_threshold = similarity_threshold
        self._setup_nltk()

    def _setup_nltk(self):
        """Инициализация NLTK ресурсов"""
        try:
            # Проверяем доступность русских стоп-слов
            self.russian_stopwords = set(stopwords.words('russian'))
            self.stemmer = SnowballStemmer('russian')
            self.use_nltk = True
        except Exception as e:
            print(f"⚠️ NLTK resources not available: {e}")
            self.russian_stopwords = set()
            self.stemmer = None
            self.use_nltk = False

    def normalize_text(self, text, sort_words=True):
        """Улучшенная нормализация текста с использованием NLTK"""
        if not isinstance(text, str): 
            return ""
            
        text = text.lower()
        
        # Базовые замены для русского языка
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
        
        if self.use_nltk:
            # Используем NLTK для более качественной обработки
            try:
                # Токенизация с помощью NLTK (лучше обрабатывает пунктуацию)
                tokens = word_tokenize(text, language='russian')
                
                # Фильтруем только буквы и цифры
                tokens = [token for token in tokens if re.match(r'^[a-zа-я0-9]+$', token)]
                
                # Удаляем русские стоп-слова
                tokens = [token for token in tokens if token not in self.russian_stopwords]
                
                # Стемминг для приведения к корневой форме
                if self.stemmer:
                    tokens = [self.stemmer.stem(token) for token in tokens]
                
                words = tokens
            except Exception as e:
                # Fallback к базовой обработке при ошибке NLTK
                text = re.sub(r'[^a-zа-я0-9\s]', ' ', text)
                words = text.split()
        else:
            # Базовая обработка без NLTK
            text = re.sub(r'[^a-zа-я0-9\s]', ' ', text)
            words = text.split()
        
        # Убираем пустые строки и короткие слова (менее 2 символов)
        words = [word for word in words if len(word.strip()) >= 2]
        
        if sort_words: 
            words = sorted(words)
        return ' '.join(words).strip()

    def normalize_brand_name(self, text):
        """Улучшенная нормализация брендового имени с NLTK"""
        base_normalized = self.normalize_text(text, sort_words=False)
        
        # Расширенный список типовых слов для более точной фильтрации
        generic_words = {
            'ооо', 'кафе', 'бар', 'ресторан', 'фастфуд', 'магазин', 'пиццерия', 
            'клуб', 'кофейня', 'столов', 'ед', 'торгов', 'центр', 'дом', 'компан',
            'предприят', 'индивидуальн', 'общеэств', 'ограничен', 'ответственност'
        }
        
        words = [word for word in base_normalized.split() if word not in generic_words]
        return ' '.join(sorted(words))

    def normalize_text_gentle(self, text):
        """Мягкая нормализация для сохранения контекста"""
        if not isinstance(text, str): 
            return ""
            
        text = text.lower().strip()
        
        # Только базовые замены
        replacements = {
            r'\bобщество с ограниченной ответственностью\b': 'ооо', 
            r'(\bип\b)|(\bиндивидуальный предприниматель\b)': 'ип',
            r'\bгород\b': 'г', 
            r'\bулица\b': 'ул', 
            r'\bдом\b': 'д', 
            r'\bстроение\b': 'стр', 
            r'\bкорпус\b': 'к',
        }
        for p, r in replacements.items(): 
            text = re.sub(p, r, text)
        
        # Убираем только явную пунктуацию, сохраняем порядок слов
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    # ==========================================================================
    # ОПРЕДЕЛЕНИЕ ЧЕТЫРЕХ НЕЗАВИСИМЫХ СУДЕЙ
    # ==========================================================================

    def get_duplicates_judge1_strict(self, df):
        """Судья 1: 'Строгий Контроль' - понижаем пороги для большей чувствительности."""
        duplicates = set()
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                name_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_name'], df.loc[id2, 'norm_name'])
                addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                # Понижаем пороги: было 95/90, стало 85/80
                if addr_sim > 85 and name_sim > 80:
                    duplicates.add(frozenset([id1, id2]))
        return duplicates

    def get_duplicates_judge2_geo(self, df):
        """Судья 2: 'Гео-Аналитик' - понижаем пороги."""
        duplicates = set()
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                # Понижаем пороги: было 97/70, стало 88/60 (включаем граничные случаи)
                if addr_sim >= 88:
                    name_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_name'], df.loc[id2, 'norm_name'])
                    if name_sim > 60:
                        duplicates.add(frozenset([id1, id2]))
        return duplicates

    def get_duplicates_judge3_brand(self, df):
        """Судья 3: 'Анализ по Бренду' - понижаем пороги."""
        duplicates = set()
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                brand_name_sim = fuzz.token_set_ratio(df.loc[id1, 'brand_name'], df.loc[id2, 'brand_name'])
                # Понижаем пороги: было 95/75, стало 85/65
                if brand_name_sim > 85:
                    addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                    if addr_sim > 65:
                        duplicates.add(frozenset([id1, id2]))
        return duplicates

    def get_duplicates_judge4_weighted(self, df):
        """Судья 4: 'Интегратор' (Взвешенная оценка) - понижаем порог."""
        duplicates = set()
        SIMILARITY_THRESHOLD = 78  # Понижаем: было 88, стало 78
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

    def get_duplicates_judge5_liberal(self, df):
        """Судья 5: 'Либеральный' - очень мягкие критерии для захвата очевидных дубликатов."""
        duplicates = set()
        ids = df.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                
                # Проверяем исходные тексты без агрессивной нормализации
                orig_name1 = str(df.loc[id1, 'orig_name']) if 'orig_name' in df.columns else ""
                orig_name2 = str(df.loc[id2, 'orig_name']) if 'orig_name' in df.columns else ""
                orig_addr1 = str(df.loc[id1, 'orig_addr']) if 'orig_addr' in df.columns else ""
                orig_addr2 = str(df.loc[id2, 'orig_addr']) if 'orig_addr' in df.columns else ""
                
                # Используем разные методы сравнения
                orig_name_ratio = fuzz.ratio(orig_name1.lower(), orig_name2.lower())
                orig_addr_ratio = fuzz.ratio(orig_addr1.lower(), orig_addr2.lower())
                
                name_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_name'], df.loc[id2, 'norm_name'])
                addr_sim = fuzz.token_set_ratio(df.loc[id1, 'norm_addr'], df.loc[id2, 'norm_addr'])
                
                # Либеральные критерии - хотя бы один должен сработать
                conditions = [
                    # Очень похожие исходные названия
                    orig_name_ratio > 85 and orig_addr_ratio > 70,
                    # Хорошие нормализованные
                    name_sim > 75 and addr_sim > 60,
                    # Отличные названия, средние адреса
                    name_sim > 90 and addr_sim > 40,
                    # Отличные адреса, средние названия
                    addr_sim > 90 and name_sim > 50,
                ]
                
                if any(conditions):
                    duplicates.add(frozenset([id1, id2]))
        
        return duplicates

    def get_duplicates_judge6_gentle_norm(self, df):
        """Судья 6: 'Мягкая нормализация' - использует менее агрессивную обработку текста."""
        duplicates = set()
        ids = df.index.tolist()
        
        # Создаем мягко нормализованные версии
        gentle_names = {}
        gentle_addrs = {}
        
        for idx in ids:
            orig_name = str(df.loc[idx, 'orig_name']) if 'orig_name' in df.columns else ""
            orig_addr = str(df.loc[idx, 'orig_addr']) if 'orig_addr' in df.columns else ""
            
            gentle_names[idx] = self.normalize_text_gentle(orig_name)
            gentle_addrs[idx] = self.normalize_text_gentle(orig_addr)
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                
                # Сравниваем мягко нормализованные тексты
                gentle_name_sim = fuzz.token_set_ratio(gentle_names[id1], gentle_names[id2])
                gentle_addr_sim = fuzz.token_set_ratio(gentle_addrs[id1], gentle_addrs[id2])
                
                # Также проверяем простое сходство строк
                simple_name_sim = fuzz.ratio(gentle_names[id1], gentle_names[id2])
                simple_addr_sim = fuzz.ratio(gentle_addrs[id1], gentle_addrs[id2])
                
                # Мягкие критерии
                conditions = [
                    # Очень похожие при token_set сравнении
                    gentle_name_sim > 80 and gentle_addr_sim > 70,
                    # Очень похожие при простом сравнении
                    simple_name_sim > 85 and simple_addr_sim > 75,
                    # Отличные названия, средние адреса
                    gentle_name_sim > 90 and gentle_addr_sim > 50,
                    # Отличные адреса, средние названия  
                    gentle_addr_sim > 90 and gentle_name_sim > 60,
                ]
                
                if any(conditions):
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
        
        # Сохраняем оригинальные индексы для дальнейшего использования
        original_indices = df.index.tolist()
        
        # Если есть колонка ID, используем её, иначе создаём свою на основе оригинальных индексов
        if id_column and id_column in df_work.columns:
            print(f"🔍 ID колонка: {id_column}")
            df_work['WorkId'] = original_indices
        else:
            df_work['WorkId'] = original_indices
        
        df_work = df_work.set_index('WorkId', drop=False)
        
        # Сохраняем оригинальные тексты для либерального судьи
        df_work['orig_name'] = df_work[name_column].apply(lambda x: str(x) if pd.notna(x) else "")
        if address_column and address_column in df_work.columns:
            df_work['orig_addr'] = df_work[address_column].apply(lambda x: str(x) if pd.notna(x) else "")
        else:
            df_work['orig_addr'] = ""
        
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
        results5 = self.get_duplicates_judge5_liberal(df_work)
        results6 = self.get_duplicates_judge6_gentle_norm(df_work)

        # Отладочная информация
        print(f"🔍 Результаты судей:")
        print(f"   Судья 1 (Строгий): {len(results1)} пар")
        print(f"   Судья 2 (Гео): {len(results2)} пар")
        print(f"   Судья 3 (Бренд): {len(results3)} пар")
        print(f"   Судья 4 (Взвешенный): {len(results4)} пар")
        print(f"   Судья 5 (Либеральный): {len(results5)} пар")
        print(f"   Судья 6 (Мягкая нормализация): {len(results6)} пар")

        # Подсчёт голосов
        all_votes = Counter(results1)
        all_votes.update(results2)
        all_votes.update(results3)
        all_votes.update(results4)
        all_votes.update(results5)
        all_votes.update(results6)

        # Анализ голосования
        vote_distribution = Counter(all_votes.values())
        print(f"📊 Распределение голосов:")
        for votes, count in sorted(vote_distribution.items()):
            print(f"   {votes} голос(ов): {count} пар")

        # Найдём пары, набравшие достаточное количество голосов
        final_duplicate_pairs = [list(pair) for pair, count in all_votes.items() if count >= min_votes]
        print(f"✅ Финальных пар дубликатов: {len(final_duplicate_pairs)} (мин. голосов: {min_votes})")
        print(f"🔍 Пары дубликатов: {final_duplicate_pairs}")

        # Группировка с использованием NetworkX
        G = nx.Graph()
        G.add_edges_from(final_duplicate_pairs)
        connected_components = list(nx.connected_components(G))
        final_groups = [list(group) for group in connected_components if len(group) > 1]
        print(f"🔍 Группы после NetworkX: {final_groups}")
        
        # Преобразуем ID обратно в индексы DataFrame
        # Поскольку мы использовали оригинальные индексы как WorkId, 
        # ID в группах уже являются правильными индексами исходного DataFrame
        final_groups_indices = final_groups
        
        print(f"🔍 Финальные группы (индексы): {final_groups_indices}")

        # Статистика
        stats = {
            "total_records": len(df),
            "duplicate_groups": len(final_groups_indices),
            "duplicate_records": sum(len(group) for group in final_groups_indices),
            "unique_records": len(df) - sum(len(group) for group in final_groups_indices),
        }

        return final_groups_indices, stats

    def create_grouped_dataframe(self, df: pd.DataFrame, duplicate_groups: List[List[int]], id_column: Optional[str] = None) -> pd.DataFrame:
        """Создание DataFrame с группировкой дубликатов"""
        grouped_df = df.copy()
        
        # Добавляем колонку для группировки дубликатов
        grouped_df['Id уникальной тт 2'] = ''
        
        print(f"🔍 ID колонка: {id_column}")
        
        # Для каждой группы дубликатов присваиваем минимальный ID из группы
        for group in duplicate_groups:
            print(f"🔍 Группа: {group}")
            if len(group) > 1:  # Проверяем, что это действительно группа дубликатов
                # Получаем настоящие ID из указанной колонки или используем индексы
                if id_column and id_column in grouped_df.columns:
                    group_ids = [grouped_df.iloc[idx][id_column] for idx in group if idx < len(grouped_df)]
                    min_id = min(group_ids) if group_ids else min(group)
                else:
                    min_id = min(group)
                
                for idx in group:
                    if idx < len(grouped_df):
                        grouped_df.iloc[idx, grouped_df.columns.get_loc('Id уникальной тт 2')] = min_id
        
        # Сортировка: сначала записи с дубликатами, потом уникальные
        grouped_df['_sort_key'] = grouped_df['Id уникальной тт 2'].apply(lambda x: 0 if x else 1)
        grouped_df = grouped_df.sort_values(['_sort_key', 'Id уникальной тт 2'])
        grouped_df = grouped_df.drop('_sort_key', axis=1)
        
        return grouped_df

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
