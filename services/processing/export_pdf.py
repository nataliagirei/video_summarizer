from fpdf import FPDF
from datetime import datetime
from pathlib import Path
import re


class PDFReporter(FPDF):
    """
    Ultimate PDF generator for audit reports and Minutes of Meeting (MoM).
    Supports UTF-8, professional HTML styling from Quill, and multilingual fonts.
    """

    def __init__(self, font_dir: str | Path, output_dir: str | Path):
        """
        Initializes the reporter with required directories.

        Args:
            font_dir: Path to the directory containing .ttf font files.
            output_dir: Path where the generated PDFs will be saved.
        """
        super().__init__()
        self.font_dir = Path(font_dir)
        self.output_dir = Path(output_dir)
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(self, title: str, html_content: str, user: str = "Natalia") -> Path:
        """
        Main method to generate a professional PDF report.
        Includes header, metadata, and formatted HTML content.
        """
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)

        # 1. Register Fonts (Unicode support for PL/RU/EN and Asian languages)
        fonts = {
            "reg": self.font_dir / "DejaVuSans.ttf",
            "bold": self.font_dir / "DejaVuSans-Bold.ttf",
            "ital": self.font_dir / "DejaVuSans-Oblique.ttf",
            "bi": self.font_dir / "DejaVuSans-BoldOblique.ttf",
            "kr": self.font_dir / "NotoSansKR-Regular.ttf"
        }

        # Detect Korean characters to switch font if necessary
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in html_content)

        font_main = "Arial"  # Default fallback
        if has_korean and fonts["kr"].exists():
            self.add_font("MainFont", "", str(fonts["kr"]), uni=True)
            font_main = "MainFont"
        elif fonts["reg"].exists():
            self.add_font("MainFont", "", str(fonts["reg"]), uni=True)
            self.add_font("MainFont", "B", str(fonts["bold"]), uni=True)
            self.add_font("MainFont", "I", str(fonts["ital"]), uni=True)
            if fonts["bi"].exists():
                self.add_font("MainFont", "BI", str(fonts["bi"]), uni=True)
            font_main = "MainFont"

        # --- Header Section ---
        self.set_font(font_main, 'B', 16)
        self.multi_cell(0, 10, title, align='C')

        self.set_font(font_main, '', 10)
        self.ln(5)
        self.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        self.cell(0, 8, f"Author: {user}", ln=True)

        # Decorative separator line
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(10)

        # --- Body Content (HTML Rendering) ---
        self.set_font(font_main, '', 11)
        self._write_formatted_html(html_content, font_main)

        # --- Save Final File ---
        # Sanitize title for filename
        safe_title = "".join(c for c in title if c.isalnum() or c in " _-").rstrip()
        file_name = f"Report_{safe_title}_{datetime.now().strftime('%H%M%S')}.pdf"
        pdf_path = self.output_dir / file_name

        self.output(str(pdf_path))
        return pdf_path

    def _write_formatted_html(self, html: str, font_name: str):
        """
        Parses HTML input from the editor and applies
        corresponding styles (Bold, Italic, Lists) to the PDF.
        """
        # 1. Pre-process tags for consistency
        html = html.replace('<ol>', '<ol_start>').replace('</ol>', '<ol_end>')
        html = html.replace('<ul>', '').replace('</ul>', '')
        html = html.replace('<p>', '').replace('</p>', '\n')
        html = html.replace('<br>', '\n').replace('<br/>', '\n')

        # 2. Split content by HTML tags using regex
        parts = re.split(r'(<[^>]+>)', html)

        is_ordered_list = False
        list_index = 1
        current_style = ""  # Tracks combined styles like Bold + Italic (BI)

        for part in parts:
            if not part: continue

            # Handle Bold
            if part in ['<b>', '<strong>']:
                if 'B' not in current_style: current_style += 'B'
                self.set_font(font_name, current_style)
            elif part in ['</b>', '</strong>']:
                current_style = current_style.replace('B', '')
                self.set_font(font_name, current_style)

            # Handle Italic
            elif part in ['<i>', '<em>']:
                if 'I' not in current_style: current_style += 'I'
                self.set_font(font_name, current_style)
            elif part in ['</i>', '<em>']:
                current_style = current_style.replace('I', '')
                self.set_font(font_name, current_style)

            # Handle Ordered and Unordered lists
            elif part == '<ol_start>':
                is_ordered_list = True
                list_index = 1
            elif part == '<ol_end>':
                is_ordered_list = False
            elif part == '<li>':
                prefix = f"\n {list_index}. " if is_ordered_list else "\n • "
                self.write(7, prefix)
                if is_ordered_list: list_index += 1
            elif part == '</li>':
                continue

            # Skip other unsupported tags
            elif part.startswith('<'):
                continue

            # Write plain text content
            else:
                # Automatic page break check for long text
                if self.get_y() > 275: self.add_page()
                self.write(7, part)