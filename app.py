import gradio as gr
import pandas as pd
from duplicate_detector import DuplicateDetector
from typing import Tuple, Optional, Dict, Any
import io

class DuplicateDetectorApp:
    def __init__(self):
        self.detector = DuplicateDetector()
        self.current_df = None
        self.duplicate_groups = None
        self.stats = None

    def load_excel_file(self, file) -> Tuple[Optional[Dict], str]:
        if file is None:
            return None, "Пожалуйста, загрузите Excel файл"
        
        try:
            self.current_df = pd.read_excel(file.name)
            
            # Удаляем безымянные колонки
            unnamed_cols = [col for col in self.current_df.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                self.current_df = self.current_df.drop(columns=unnamed_cols)
            
            # Автоматически находим нужные колонки
            name_col = self.detector.find_name_column(self.current_df)
            address_col = self.detector.find_address_column(self.current_df)
            
            if not name_col:
                return None, "❌ Не найдена колонка с названиями. Убедитесь, что в файле есть колонка содержащая 'название', 'наименование' или 'имя'"

            if not address_col:
                return None, "❌ Не найдена колонка с адресами. Убедитесь, что в файле есть колонка содержащая 'адрес', 'address' или 'address'"
            
            info = f"✅ **Файл загружен успешно!**\n\n"
            info += f"- Всего записей: **{len(self.current_df)}**\n"
            info += f"- Колонка названий: **{name_col}**\n"
            info += f"- Колонка адресов: **{address_col if address_col else 'не найдена'}**\n\n"
            info += "🔍 Нажмите **'Найти дубликаты'** для начала проверки"
            
            return (
                {
                    "data": self.current_df.values.tolist(),
                    "headers": self.current_df.columns.tolist(),
                },
                info
            )
        
        except Exception as e:
            error_msg = f"❌ **Ошибка загрузки файла**: {str(e)}"
            return None, error_msg

    def find_duplicates(self) -> Tuple[Optional[Dict], str]:
        if self.current_df is None:
            return None, "❌ Сначала загрузите Excel файл"
        
        try:
            # Фиксированная точность 70% (для комбинированного метода)
            self.detector.similarity_threshold = 0.70
            
            # Автоматически находим нужные колонки
            name_col = self.detector.find_name_column(self.current_df)
            address_col = self.detector.find_address_column(self.current_df)
            
            if not name_col:
                return None, "❌ Не найдена колонка с названиями"
            
            self.duplicate_groups, self.stats = self.detector.find_duplicates(
                df=self.current_df,
                name_column=name_col,
                address_column=address_col
            )
            
            if self.duplicate_groups:
                styled_data = self.detector.create_styled_dataframe(
                    self.current_df, 
                    self.duplicate_groups,
                    True  # используем темную тему для лучшей видимости
                )
                
                report = f"🔍 **Результаты поиска дубликатов:**\n\n"
                report += f"- Всего записей: **{self.stats['total_records']}**\n"
                report += f"- Найдено групп дубликатов: **{self.stats['duplicate_groups']}**\n"
                report += f"- Записей-дубликатов: **{self.stats['duplicate_records']}**\n"
                report += f"- Уникальных записей: **{self.stats['unique_records']}**\n\n"
                report += f"💡 **Дубликаты сгруппированы и выделены цветом**\n"
                report += f"⚙️ Метод: **Комбинированный алгоритм (70%)**\n"
                report += f"🔬 **Улучшенный алгоритм** - строковое сходство + Жаккар + нормализация"
                
                return styled_data, report
            else:
                return (
                    {
                        "data": self.current_df.values.tolist(),
                        "headers": self.current_df.columns.tolist(),
                    },
                    f"✅ **Дубликаты не найдены!**\n\nВсе {len(self.current_df)} записей уникальны при использовании комбинированного алгоритма (70%)"
                )
        
        except Exception as e:
            return None, f"❌ **Ошибка поиска дубликатов**: {str(e)}"

    def download_results(self):
        if self.current_df is None or self.duplicate_groups is None:
            return
        
        try:
            grouped_df = self.detector.create_grouped_dataframe(self.current_df, self.duplicate_groups)
            
            output_file = "duplicates_result.xlsx"
            grouped_df.to_excel(output_file, index=False)
            
            print(f"✅ Файл сохранен: {output_file}")
        
        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")

    def create_interface(self):
        with gr.Blocks(
            title="🔍 Поиск дубликатов в Excel",
            theme=gr.themes.Soft(),
            css="""
            .main-header {
                text-align: center;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 1.5rem;
            }
            .button-row {
                display: flex;
                gap: 1rem;
                justify-content: center;
                margin: 1rem 0;
            }
            """
        ) as app:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🔍 Поиск дубликатов в Excel</h1>
                <p>Загрузите Excel файл и найдите дубликаты по названиям и адресам</p>
            </div>
            """)
            
            # Упрощенный интерфейс - только загрузка файла и кнопки
            file_input = gr.File(
                label="📁 Загрузить Excel файл",
                file_types=[".xlsx", ".xls"],
                file_count="single"
            )
            
            file_info = gr.Markdown("Файл не загружен")
            
            with gr.Row():
                find_btn = gr.Button(
                    "🔍 Найти дубликаты", 
                    variant="primary",
                    size="lg",
                    scale=1
                )
                download_btn = gr.Button(
                    "💾 Скачать результат",
                    variant="secondary",
                    size="lg",
                    visible=False,
                    scale=1
                )
            
            # Таблица с результатами внизу
            results_table = gr.DataFrame(
                label="📊 Результаты",
                interactive=False,
                wrap=True,
                max_height=600
            )
            
            # Обработчики событий
            file_input.upload(
                fn=self.load_excel_file,
                inputs=[file_input],
                outputs=[results_table, file_info]
            )
            
            find_btn.click(
                fn=lambda: (*self.find_duplicates(), gr.update(visible=True)),
                inputs=[],
                outputs=[results_table, file_info, download_btn]
            )
            
            download_btn.click(
                fn=self.download_results,
                inputs=[],
                outputs=[]
            )
            
        return app

def main():
    app = DuplicateDetectorApp()
    interface = app.create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )

if __name__ == "__main__":
    main() 