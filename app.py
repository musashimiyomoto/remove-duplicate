import gradio as gr
import pandas as pd
from duplicate_detector import DuplicateDetector
from typing import Tuple, Optional, Dict


class DuplicateDetectorApp:
    def __init__(self):
        self.detector = DuplicateDetector()
        self.current_df = None
        self.duplicate_groups = None
        self.stats = None

    def load_excel_file(self, file) -> Tuple[Optional[Dict], str]:
        if file is None:
            return None, "Please upload an Excel file"

        try:
            self.current_df = pd.read_excel(file.name)

            unnamed_cols = [
                col for col in self.current_df.columns if col.startswith("Unnamed")
            ]
            if unnamed_cols:
                self.current_df = self.current_df.drop(columns=unnamed_cols)

            name_col = self.detector.find_name_column(self.current_df)
            address_col = self.detector.find_address_column(self.current_df)

            if not name_col:
                return (
                    None,
                    "❌ Name column not found. Make sure the file has a column containing 'name', 'title', or 'company'",
                )

            if not address_col:
                return (
                    None,
                    "❌ Address column not found. Make sure the file has a column containing 'address' or 'location'",
                )

            info = f"✅ **File loaded successfully!**\n\n"
            info += f"- Total records: **{len(self.current_df)}**\n"
            info += f"- Name column: **{name_col}**\n"
            info += f"- Address column: **{address_col if address_col else 'not found'}**\n\n"
            info += "🔍 Click **'Find Duplicates'** to start checking"

            return (
                {
                    "data": self.current_df.values.tolist(),
                    "headers": self.current_df.columns.tolist(),
                },
                info,
            )

        except Exception as e:
            error_msg = f"❌ **File loading error**: {str(e)}"
            return None, error_msg

    def find_duplicates(self) -> Tuple[Optional[Dict], str]:
        if self.current_df is None:
            return None, "❌ Please upload an Excel file first"

        try:
            self.detector.similarity_threshold = 0.75

            name_col = self.detector.find_name_column(self.current_df)
            address_col = self.detector.find_address_column(self.current_df)

            if not name_col:
                return None, "❌ Name column not found"

            self.duplicate_groups, self.stats = self.detector.find_duplicates(
                df=self.current_df, name_column=name_col, address_column=address_col
            )

            if self.duplicate_groups:
                styled_data = self.detector.create_styled_dataframe(
                    self.current_df,
                    self.duplicate_groups,
                    True,
                )

                report = f"🔍 **Duplicate search results:**\n\n"
                report += f"- Total records: **{self.stats['total_records']}**\n"
                report += (
                    f"- Duplicate groups found: **{self.stats['duplicate_groups']}**\n"
                )
                report += (
                    f"- Duplicate records: **{self.stats['duplicate_records']}**\n"
                )
                report += f"- Unique records: **{self.stats['unique_records']}**\n\n"
                report += f"💡 **Duplicates grouped and color-highlighted**\n"
                report += f"⚙️ Method: **Enhanced algorithm with address verification (75%)**\n"
                report += f"🔬 **Smart algorithm** - considers house numbers, flexible similarity requirements"

                return styled_data, report
            else:
                return (
                    {
                        "data": self.current_df.values.tolist(),
                        "headers": self.current_df.columns.tolist(),
                    },
                    f"✅ **No duplicates found!**\n\nAll {len(self.current_df)} records are unique using enhanced algorithm (75%)",
                )

        except Exception as e:
            return None, f"❌ **Duplicate search error**: {str(e)}"

    def download_results(self):
        if self.current_df is None or self.duplicate_groups is None:
            return

        try:
            grouped_df = self.detector.create_grouped_dataframe(
                self.current_df, self.duplicate_groups
            )

            output_file = "duplicates_result.xlsx"
            grouped_df.to_excel(output_file, index=False)

            print(f"✅ File saved: {output_file}")

        except Exception as e:
            print(f"❌ File saving error: {e}")

    def create_interface(self):
        with gr.Blocks(
            title="🔍 Excel Duplicate Finder",
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
            """,
        ) as app:

            gr.HTML(
                """
            <div class="main-header">
                <h1>🔍 Excel Duplicate Finder</h1>
                <p>Upload an Excel file and find duplicates by names and addresses</p>
            </div>
            """
            )

            file_input = gr.File(
                label="📁 Upload Excel file",
                file_types=[".xlsx", ".xls"],
                file_count="single",
            )

            file_info = gr.Markdown("File not loaded")

            with gr.Row():
                find_btn = gr.Button(
                    "🔍 Find Duplicates", variant="primary", size="lg", scale=1
                )
                download_btn = gr.Button(
                    "💾 Download Results",
                    variant="secondary",
                    size="lg",
                    visible=False,
                    scale=1,
                )

            results_table = gr.DataFrame(
                label="📊 Results", interactive=False, wrap=True, max_height=600
            )

            file_input.upload(
                fn=self.load_excel_file,
                inputs=[file_input],
                outputs=[results_table, file_info],
            )

            find_btn.click(
                fn=lambda: (*self.find_duplicates(), gr.update(visible=True)),
                inputs=[],
                outputs=[results_table, file_info, download_btn],
            )

            download_btn.click(fn=self.download_results, inputs=[], outputs=[])

        return app


def main():
    app = DuplicateDetectorApp()
    interface = app.create_interface()
    interface.launch(
        share=False, server_name="0.0.0.0", server_port=7860, show_api=False
    )


if __name__ == "__main__":
    main()
