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
        if file is None:
            return None, "", "", "Please upload an Excel file"
        
        try:
            self.current_df = pd.read_excel(file.name)
            
            unnamed_cols = [col for col in self.current_df.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                self.current_df = self.current_df.drop(columns=unnamed_cols)
            
            info = f"📊 **File loaded successfully!**\n\n"
            info += f"- Total records: **{len(self.current_df)}**\n"
            info += f"- Number of columns: **{len(self.current_df.columns)}**\n"
            
            try:
                file_size = getattr(file, 'size', None)
                if file_size is not None:
                    info += f"- File size: **{file_size} bytes**\n"
            except:
                pass
            
            columns = self.current_df.columns.tolist()
            name_column_choices = gr.Dropdown(choices=columns, value=columns[0] if columns else None)
            address_column_choices = gr.Dropdown(choices=["Don't use"] + columns, value="Don't use")
            
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
            error_msg = f"❌ **Error loading file**: {str(e)}"
            return None, error_msg, "", ""

    def find_duplicates(self, name_column: str, address_column: str, threshold: int, is_dark_theme: bool = False) -> Tuple[Optional[Dict], str]:
        if self.current_df is None:
            return None, "❌ Please upload an Excel file first"
        
        if not name_column:
            return None, "❌ Please select a column for duplicate search"
        
        try:
            self.detector.similarity_threshold = threshold
            
            address_col = address_column if address_column != "Don't use" else None
            
            self.duplicate_groups, self.stats = self.detector.find_duplicates(
                df=self.current_df,
                name_column=name_column,
                address_column=address_col
            )
            
            if self.duplicate_groups:
                styled_data = self.detector.create_styled_dataframe(
                    self.current_df, 
                    self.duplicate_groups,
                    is_dark_theme
                )
                
                theme_text = "dark theme" if is_dark_theme else "light theme"
                report = f"🔍 **Duplicate search results ({theme_text}):**\n\n"
                report += f"- Total records: **{self.stats['total_records']}**\n"
                report += f"- Duplicate groups found: **{self.stats['duplicate_groups']}**\n"
                report += f"- Duplicate records: **{self.stats['duplicate_records']}**\n"
                report += f"- Unique records: **{self.stats['unique_records']}**\n\n"
                report += f"💡 **Duplicates are grouped together and highlighted with colors optimized for {theme_text}**\n"
                report += f"⚙️ Similarity threshold: **{threshold}%**"
                
                return styled_data, report
            else:
                return (
                    {
                        "data": self.current_df.values.tolist(),
                        "headers": self.current_df.columns.tolist(),
                    },
                    f"✅ **No duplicates found!**\n\nAll {len(self.current_df)} records are unique at {threshold}% similarity threshold"
                )
        
        except Exception as e:
            return None, f"❌ **Error searching for duplicates**: {str(e)}"

    def download_results(self) -> Optional[str]:
        if self.current_df is None or self.duplicate_groups is None:
            return None
        
        try:
            grouped_df = self.detector.create_grouped_dataframe(self.current_df, self.duplicate_groups)
            
            output_file = "duplicates_result.xlsx"
            grouped_df.to_excel(output_file, index=False)
            
            return output_file
        
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def create_interface(self):
        with gr.Blocks(
            title="🔍 Excel Duplicate Detector",
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
            
            gr.HTML("""
            <div class="main-header">
                <h1>🔍 Excel Duplicate Detector</h1>
                <p>Upload an Excel file and find duplicate records with smart highlighting and grouping</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>📋 Settings</h3>")
                    
                    file_input = gr.File(
                        label="📁 Upload Excel file",
                        file_types=[".xlsx", ".xls"],
                        file_count="single"
                    )
                    
                    file_info = gr.Markdown("No file uploaded")
                    
                    with gr.Group():
                        name_column = gr.Dropdown(
                            label="🏷️ Column for duplicate search (name)",
                            choices=[],
                            interactive=True
                        )
                        
                        address_column = gr.Dropdown(
                            label="📍 Address column (optional)",
                            choices=["Don't use"],
                            value="Don't use",
                            interactive=True
                        )
                        
                        threshold = gr.Slider(
                            label="🎯 Similarity threshold (%)",
                            minimum=50,
                            maximum=100,
                            value=85,
                            step=5,
                            info="Higher values mean stricter duplicate detection"
                        )
                        
                        dark_theme_toggle = gr.Checkbox(
                            label="🌙 Dark theme colors",
                            value=True,
                            info="Enable if you're using dark theme for better color visibility",
                            visible=False,
                        )
                    
                    with gr.Row():
                        find_btn = gr.Button(
                            "🔍 Find Duplicates", 
                            variant="primary",
                            size="lg"
                        )
                        download_btn = gr.Button(
                            "💾 Download Results",
                            variant="secondary",
                            visible=False
                        )
                
                with gr.Column(scale=2):
                    gr.HTML("<h3>📊 Results</h3>")
                    
                    results_info = gr.Markdown("Upload a file to get started")
                    
                    data_table = gr.Dataframe(
                        label="Data with highlighted and grouped duplicates",
                        interactive=False,
                        wrap=True
                    )
                    
                    download_file = gr.File(
                        label="Download results",
                        visible=False
                    )
            
            file_input.upload(
                fn=self.load_excel_file,
                inputs=[file_input],
                outputs=[data_table, file_info, name_column, address_column]
            )
            
            find_btn.click(
                fn=self.find_duplicates,
                inputs=[name_column, address_column, threshold, dark_theme_toggle],
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