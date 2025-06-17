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

    def load_excel_file(self, file) -> Tuple[Optional[Dict], str, str, str]:
        """Загружает Excel файл и отображает его содержимое"""
        if file is None:
            return None, "", "", "Пожалуйста, загрузите файл Excel"
        
        try:
            # Читаем Excel файл
            self.current_df = pd.read_excel(file.name)
            
            # Удаляем колонки "Unnamed" (пустые колонки)
            unnamed_cols = [col for col in self.current_df.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                self.current_df = self.current_df.drop(columns=unnamed_cols)
            
            # Получаем информацию о файле
            info = f"📊 **Файл загружен успешно!**\n\n"
            info += f"- Количество записей: **{len(self.current_df)}**\n"
            info += f"- Количество колонок: **{len(self.current_df.columns)}**\n"
            
            # Пытаемся получить размер файла, если доступно
            try:
                file_size = getattr(file, 'size', None)
                if file_size is not None:
                    info += f"- Размер файла: **{file_size} байт**\n"
            except:
                pass
            
            # Создаем список колонок для выпадающих списков
            columns = self.current_df.columns.tolist()
            name_column_choices = gr.Dropdown(choices=columns, value=columns[0] if columns else None)
            address_column_choices = gr.Dropdown(choices=["Не использовать"] + columns, value="Не использовать")
            
            # Возвращаем данные для отображения
            return (
                {
                    "data": self.current_df.values.tolist(),
                    "headers": self.current_df.columns.tolist(),
                },
                info,
                name_column_choices,
                address_column_choices
            )
        
        except Exception as e:
            error_msg = f"❌ **Ошибка при загрузке файла**: {str(e)}"
            return None, error_msg, "", ""

    def find_duplicates(self, name_column: str, address_column: str, threshold: int) -> Tuple[Optional[Dict], str]:
        """Находит дубликаты и отображает их с подсветкой"""
        if self.current_df is None:
            return None, "❌ Сначала загрузите файл Excel"
        
        if not name_column:
            return None, "❌ Выберите колонку для поиска дубликатов"
        
        try:
            # Обновляем порог похожести
            self.detector.similarity_threshold = threshold
            
            # Определяем колонку адреса
            address_col = address_column if address_column != "Не использовать" else None
            
            # Ищем дубликаты
            self.duplicate_groups, self.stats = self.detector.find_duplicates(
                df=self.current_df,
                name_column=name_column,
                address_column=address_col
            )
            
            # Создаем стилизованную таблицу
            if self.duplicate_groups:
                styled_data = self.detector.create_styled_dataframe(
                    self.current_df, 
                    self.duplicate_groups
                )
                
                # Формируем отчет
                report = f"🔍 **Результаты поиска дубликатов:**\n\n"
                report += f"- Всего записей: **{self.stats['total_records']}**\n"
                report += f"- Найдено групп дубликатов: **{self.stats['duplicate_groups']}**\n"
                report += f"- Дублирующихся записей: **{self.stats['duplicate_records']}**\n"
                report += f"- Уникальных записей: **{self.stats['unique_records']}**\n\n"
                report += f"💡 **Дубликаты выделены разными цветами по группам**\n"
                report += f"⚙️ Порог похожести: **{threshold}%**"
                
                return styled_data, report
            else:
                return (
                    {
                        "data": self.current_df.values.tolist(),
                        "headers": self.current_df.columns.tolist(),
                    },
                    f"✅ **Дубликаты не найдены!**\n\nВсе {len(self.current_df)} записей уникальны при пороге похожести {threshold}%"
                )
        
        except Exception as e:
            return None, f"❌ **Ошибка при поиске дубликатов**: {str(e)}"

    def download_results(self) -> Optional[str]:
        """Сохраняет результаты в Excel файл для скачивания"""
        if self.current_df is None or self.duplicate_groups is None:
            return None
        
        try:
            # Создаем копию DataFrame
            df_result = self.current_df.copy()
            
            # Добавляем колонку с группами дубликатов
            df_result['Группа_дубликатов'] = 0
            
            for group_idx, group in enumerate(self.duplicate_groups, 1):
                for row_idx in group:
                    df_result.loc[row_idx, 'Группа_дубликатов'] = group_idx
            
            # Сохраняем в файл
            output_file = "duplicates_result.xlsx"
            df_result.to_excel(output_file, index=False)
            
            return output_file
        
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}")
            return None

    def create_interface(self):
        """Создает интерфейс Gradio"""
        with gr.Blocks(
            title="🔍 Детектор дубликатов в Excel",
            theme=gr.themes.Soft(),
            css="""
            .main-header {
                text-align: center;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .stats-box {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            """
        ) as app:
            
            # Заголовок
            gr.HTML("""
            <div class="main-header">
                <h1>🔍 Детектор дубликатов в Excel файлах</h1>
                <p>Загрузите Excel файл и найдите дублирующиеся записи с умной подсветкой</p>
            </div>
            """)
            
            # Основное содержимое
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>📋 Настройки</h3>")
                    
                    # Загрузка файла
                    file_input = gr.File(
                        label="📁 Загрузить Excel файл",
                        file_types=[".xlsx", ".xls"],
                        file_count="single"
                    )
                    
                    # Информация о файле и настройки
                    file_info = gr.Markdown("Файл не загружен")
                    
                    with gr.Group():
                        name_column = gr.Dropdown(
                            label="🏷️ Колонка для поиска дубликатов (название)",
                            choices=[],
                            interactive=True
                        )
                        
                        address_column = gr.Dropdown(
                            label="📍 Колонка с адресами (опционально)",
                            choices=["Не использовать"],
                            value="Не использовать",
                            interactive=True
                        )
                        
                        threshold = gr.Slider(
                            label="🎯 Порог похожести (%)",
                            minimum=50,
                            maximum=100,
                            value=85,
                            step=5,
                            info="Чем выше значение, тем строже поиск дубликатов"
                        )
                    
                    # Кнопки
                    with gr.Row():
                        find_btn = gr.Button(
                            "🔍 Найти дубликаты", 
                            variant="primary",
                            size="lg"
                        )
                        download_btn = gr.Button(
                            "💾 Скачать результат",
                            variant="secondary",
                            visible=False
                        )
                
                with gr.Column(scale=2):
                    gr.HTML("<h3>📊 Результаты</h3>")
                    
                    # Отчет о результатах
                    results_info = gr.Markdown("Загрузите файл для начала работы")
                    
                    # Таблица с данными
                    data_table = gr.Dataframe(
                        label="Данные с выделенными дубликатами",
                        interactive=False,
                        wrap=True
                    )
                    
                    # Файл для скачивания
                    download_file = gr.File(
                        label="Результат для скачивания",
                        visible=False
                    )
            
            # Подвал с информацией
            gr.HTML("""
            <div style="margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; text-align: center;">
                <h4>💡 Как это работает?</h4>
                <p><strong>1.</strong> Загрузите Excel файл с данными</p>
                <p><strong>2.</strong> Выберите колонку для поиска дубликатов</p>
                <p><strong>3.</strong> При желании укажите колонку с адресами для более точного поиска</p>
                <p><strong>4.</strong> Настройте порог похожести и нажмите "Найти дубликаты"</p>
                <p><strong>5.</strong> Дубликаты будут выделены разными цветами в таблице</p>
            </div>
            """)
            
            # События
            file_input.upload(
                fn=self.load_excel_file,
                inputs=[file_input],
                outputs=[data_table, file_info, name_column, address_column]
            )
            
            find_btn.click(
                fn=self.find_duplicates,
                inputs=[name_column, address_column, threshold],
                outputs=[data_table, results_info]
            ).then(
                fn=lambda: [gr.update(visible=True), gr.update(visible=True)],
                outputs=[download_btn, download_file]
            )
            
            download_btn.click(
                fn=self.download_results,
                outputs=[download_file]
            )
        
        return app

def main():
    app_instance = DuplicateDetectorApp()
    app = app_instance.create_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main() 