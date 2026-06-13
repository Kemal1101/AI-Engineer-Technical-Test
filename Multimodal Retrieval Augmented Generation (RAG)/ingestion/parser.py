# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF
# pyrefly: ignore [missing-import]
import pdfplumber
import io
import base64
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage

class DocumentParser:
    def __init__(self):
        # We use gemini-2.5-flash for vision tasks
        self.vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    def extract_text_and_images(self, pdf_path):
        doc = fitz.open(pdf_path)
        extracted_data = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                extracted_data.append({"type": "text", "content": text.strip(), "page": page_num + 1})

            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image for description
                image = Image.open(io.BytesIO(image_bytes))
                
                # Filter small icons/logos if necessary based on size, but let's describe all for now or filter small ones
                if image.width < 100 or image.height < 100:
                    continue

                description = self._describe_image(image_bytes)
                if description:
                    extracted_data.append({"type": "image_description", "content": f"[Deskripsi Grafik]: {description}", "page": page_num + 1})

        doc.close()
        return extracted_data

    def _describe_image(self, image_bytes: bytes) -> str:
        prompt = "Kamu adalah analis keuangan. Jelaskan secara detail isi dari grafik/chart laporan keuangan berikut, termasuk angka penting, tren, dan kesimpulannya dalam bentuk teks naratif. Jika gambar ini bukan grafik atau bukan tabel (hanya logo/ornamen), balas dengan 'SKIP'."
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ]
            )
            response = self.vision_model.invoke([message])
            content = response.content.strip()
            if "SKIP" in content.upper() and len(content) < 15:
                return ""
            return content
        except Exception as e:
            print(f"Error describing image: {e}")
            return ""

    def extract_tables_as_markdown(self, pdf_path):
        extracted_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    md_table = self._table_to_markdown(table)
                    if md_table:
                        extracted_data.append({"type": "table", "content": md_table, "page": i + 1})
        return extracted_data

    def _table_to_markdown(self, table):
        if not table or not table[0]:
            return None
        
        md_lines = []
        # Header
        header = table[0]
        md_lines.append("| " + " | ".join([str(h).replace('\n', ' ') if h else "" for h in header]) + " |")
        # Separator
        md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
        # Rows
        for row in table[1:]:
            md_lines.append("| " + " | ".join([str(c).replace('\n', ' ') if c else "" for c in row]) + " |")
        
        return "\n".join(md_lines)
        
    def parse_document(self, pdf_path):
        print(f"Parsing {pdf_path}...")
        data = self.extract_text_and_images(pdf_path)
        tables = self.extract_tables_as_markdown(pdf_path)
        data.extend(tables)
        # Sort by page
        data.sort(key=lambda x: x['page'])
        return data
