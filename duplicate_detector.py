import pandas as pd
from unidecode import unidecode
from fuzzywuzzy import fuzz
import re
from typing import List, Dict, Tuple, Optional
import colorsys

class DuplicateDetector:
    def __init__(self, similarity_threshold: int = 85):
        self.similarity_threshold = similarity_threshold
        
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
        
    def generate_colors(self, num_colors: int, is_dark_theme: bool = False) -> List[str]:
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
        
    def normalize_text(self, text: str, remove_org_forms: bool = True, remove_venue_types: bool = True) -> str:
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

    def transliterate_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        try:
            transliterated = unidecode(str(text))
            return self.normalize_text(transliterated)
        except:
            return self.normalize_text(str(text))

    def normalize_address(self, address: str) -> str:
        if pd.isna(address):
            return ""
        
        address = str(address).lower().strip()
        address = re.sub(self.common_words['postal_codes'], '', address)
        
        for pattern, replacement in self.common_words['address_abbreviations'].items():
            address = re.sub(pattern, replacement, address)
        
        address = re.sub(r'[,.\-№]', ' ', address)
        address = re.sub(self.common_words['extra_spaces'], ' ', address).strip()
        
        return address

    def calculate_similarity(self, text1: str, text2: str, address1: Optional[str] = None, address2: Optional[str] = None) -> float:
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

    def find_duplicates(self, df: pd.DataFrame, name_column: str, address_column: Optional[str] = None, id_column: Optional[str] = None) -> Tuple[List[List[int]], Dict]:
        duplicate_groups = []
        processed_indices = set()
        
        for i in range(len(df)):
            if i in processed_indices:
                continue
                
            current_group = [i]
            current_name = df.iloc[i][name_column]
            current_address = df.iloc[i][address_column] if address_column and address_column in df.columns else None
            
            for j in range(i + 1, len(df)):
                if j in processed_indices:
                    continue
                
                compare_name = df.iloc[j][name_column]
                compare_address = df.iloc[j][address_column] if address_column and address_column in df.columns else None
                
                similarity = self.calculate_similarity(
                    current_name, compare_name, 
                    current_address, compare_address
                )
                
                if similarity >= self.similarity_threshold:
                    current_group.append(j)
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                processed_indices.update(current_group)
        
        stats = {
            'total_records': len(df),
            'duplicate_groups': len(duplicate_groups),
            'duplicate_records': sum(len(group) for group in duplicate_groups),
            'unique_records': len(df) - sum(len(group) for group in duplicate_groups)
        }
        
        return duplicate_groups, stats

    def create_grouped_dataframe(self, df: pd.DataFrame, duplicate_groups: List[List[int]]) -> pd.DataFrame:
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
        
        remaining_indices = set(range(len(df))) - set(idx for group in duplicate_groups for idx in group)
        for row_idx in remaining_indices:
            row_data = df.iloc[row_idx].to_dict()
            row_data['Duplicate_Group'] = 0
            grouped_data.append(row_data)
        
        result_df = pd.DataFrame(grouped_data)
        result_df = result_df.sort_values(['Duplicate_Group', result_df.columns[0]], ascending=[False, True])
        
        return result_df

    def create_styled_dataframe(self, df: pd.DataFrame, duplicate_groups: List[List[int]], is_dark_theme: bool = False) -> Dict:
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