import pandas as pd
from unidecode import unidecode
from fuzzywuzzy import fuzz
import re
import os

class DuplicateDetector:
    def __init__(self, similarity_threshold=85):
        self.similarity_threshold = similarity_threshold
        self.duplicate_groups = []
        
        self.common_words = {
            'org_forms': r'\b(ип|ооо|оао|зао|тов|ltd|llc|inc|corporation|corp|company|co)\b',
            'venue_types': r'\b(кафе|ресторан|бар|столовая|магазин|торговая точка|тт|cafe|restaurant|bar|shop|store)\b',
            'punctuation': r'[.,;:!?()"\'\-\№#@$%^&*+={}|\\`~<>/]',
            'extra_spaces': r'\s+',
            'postal_codes': r'^\d{5,6},?\s*',
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
        
    def normalize_text(self, text, remove_org_forms=True, remove_venue_types=True):
        if pd.isna(text):
            return ""
        
        text = str(text).strip().lower()
        text = re.sub(self.common_words['extra_spaces'], ' ', text)
        text = re.sub(self.common_words['punctuation'], ' ', text)
        
        if remove_org_forms:
            text = re.sub(self.common_words['org_forms'], '', text)
        
        if remove_venue_types:
            text = re.sub(self.common_words['venue_types'], '', text)
        
        text = re.sub(self.common_words['extra_spaces'], ' ', text).strip()
        return text

    def transliterate_text(self, text):
        if pd.isna(text):
            return ""
        
        try:
            transliterated = unidecode(str(text))
            return self.normalize_text(transliterated)
        except:
            return self.normalize_text(str(text))

    def normalize_address(self, address):
        if pd.isna(address):
            return ""
        
        address = str(address).lower().strip()
        address = re.sub(self.common_words['postal_codes'], '', address)
        
        for pattern, replacement in self.common_words['address_abbreviations'].items():
            address = re.sub(pattern, replacement, address)
        
        address = re.sub(r'[,.\-№]', ' ', address)
        address = re.sub(self.common_words['extra_spaces'], ' ', address).strip()
        
        return address

    def calculate_similarity(self, text1, text2, address1=None, address2=None):
        if pd.isna(text1) or pd.isna(text2):
            return 0
        
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        trans1 = self.transliterate_text(text1)
        trans2 = self.transliterate_text(text2)
        
        similarities = []
        
        if norm1 and norm2:
            similarities.extend([
                fuzz.ratio(norm1, norm2),
                fuzz.partial_ratio(norm1, norm2),
                fuzz.token_sort_ratio(norm1, norm2),
                fuzz.token_set_ratio(norm1, norm2)
            ])
        
        if trans1 and trans2:
            similarities.extend([
                fuzz.ratio(trans1, trans2),
                fuzz.partial_ratio(trans1, trans2),
                fuzz.token_sort_ratio(trans1, trans2),
                fuzz.token_set_ratio(trans1, trans2)
            ])
        
        if norm1 and trans2:
            similarities.extend([
                fuzz.ratio(norm1, trans2),
                fuzz.token_sort_ratio(norm1, trans2)
            ])
        
        if trans1 and norm2:
            similarities.extend([
                fuzz.ratio(trans1, norm2),
                fuzz.token_sort_ratio(trans1, norm2)
            ])
        
        if address1 and address2:
            addr1_norm = self.normalize_address(address1)
            addr2_norm = self.normalize_address(address2)
            
            if addr1_norm and addr2_norm:
                addr_similarity = fuzz.ratio(addr1_norm, addr2_norm)
                if addr_similarity > 80:
                    similarities = [min(s * 1.15, 100) for s in similarities]
                
                similarities.append(addr_similarity * 0.8)
        
        return max(similarities) if similarities else 0

    def find_duplicates(self, df, name_column='Название ТТ', address_column='Адрес ТТ', id_column='Id'):
        print(f"Поиск дубликатов в {len(df)} записях...")
        
        duplicate_groups = []
        processed_indices = set()
        
        for i in range(len(df)):
            if i in processed_indices:
                continue
                
            current_group = [i]
            current_name = df.iloc[i][name_column]
            current_address = df.iloc[i][address_column] if address_column in df.columns else None
            
            for j in range(i + 1, len(df)):
                if j in processed_indices:
                    continue
                
                compare_name = df.iloc[j][name_column]
                compare_address = df.iloc[j][address_column] if address_column in df.columns else None
                
                similarity = self.calculate_similarity(
                    current_name, compare_name, 
                    current_address, compare_address
                )
                
                if similarity >= self.similarity_threshold:
                    current_group.append(j)
            
            if len(current_group) > 1:
                group_data = []
                for idx in current_group:
                    group_data.append({
                        'index': idx,
                        'id': df.iloc[idx][id_column],
                        'name': df.iloc[idx][name_column],
                        'address': df.iloc[idx][address_column] if address_column in df.columns else None
                    })
                
                duplicate_groups.append(group_data)
                processed_indices.update(current_group)
                
                print(f"Найдена группа дубликатов ({len(current_group)} записей):")
                for item in group_data:
                    print(f"  - ID: {item['id']}, Название: {item['name']}")
        
        print(f"\nВсего найдено групп дубликатов: {len(duplicate_groups)}")
        return duplicate_groups

    def mark_duplicates(self, df, duplicate_groups, unique_id_column='Id уникальной тт 2'):
        df_copy = df.copy()
        
        if unique_id_column not in df_copy.columns:
            df_copy[unique_id_column] = None
        
        for group in duplicate_groups:
            min_id = min(item['id'] for item in group)
            
            for item in group:
                df_copy.loc[item['index'], unique_id_column] = min_id
        
        return df_copy

def main():
    input_file = "Sample-for-Agentsify.xlsx"
    output_file = "Sample-for-Agentsify-Deduplicated.xlsx"
    
    if not os.path.exists(input_file):
        print(f"Файл не найден: {input_file}")
        return
    
    print(f"Читаем файл: {input_file}")
    df = pd.read_excel(input_file)
    
    print(f"Загружено записей: {len(df)}")
    print(f"Колонки: {list(df.columns)}")
    
    detector = DuplicateDetector(similarity_threshold=80)
    
    duplicate_groups = detector.find_duplicates(
        df, 
        name_column='Название ТТ',
        address_column='Адрес ТТ',
        id_column='Id'
    )
    
    if duplicate_groups:
        df_marked = detector.mark_duplicates(df, duplicate_groups)
        
        df_marked.to_excel(output_file, index=False)
        print(f"\nРезультат сохранен в: {output_file}")
        
        marked_count = df_marked['Id уникальной тт 2'].notna().sum()
        print(f"Отмечено записей как дубликаты: {marked_count}")
        print(f"Групп дубликатов: {len(duplicate_groups)}")
        
    else:
        print("Дубликаты не найдены")

if __name__ == "__main__":
    main()
