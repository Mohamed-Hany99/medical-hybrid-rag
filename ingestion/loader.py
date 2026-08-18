import os
import base64
import pymupdf  # Updated from fitz
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN")
# موديل الـ Vision المخصص للـ OCR
HF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "google/gemma-4-26B-A4B-it")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

OCR_TEXT_THRESHOLD = int(os.getenv("OCR_TEXT_THRESHOLD", "50"))
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "1024"))

# ============================================================
# Data Structures
# ============================================================
@dataclass
class PageRecord:
    document_name: str
    page_number: int
    text: str
    extraction_method: str
    ocr_status: str = "N/A"

# ============================================================
# Hugging Face OCR Engine
# ============================================================
def create_hf_client() -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing from .env")
    return InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)

def page_to_base64_image(page: pymupdf.Page) -> str:
    """Render a PDF page into PNG in memory and encode it as base64."""
    pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("utf-8")

def ocr_page(client: InferenceClient, page: pymupdf.Page) -> str:
    """OCR a single PDF page through Hugging Face Vision model."""
    image_b64 = page_to_base64_image(page)
    image_url = f"data:image/png;base64,{image_b64}"

    prompt = """
    Perform OCR on this document page.
    Extract all visible text accurately.
    Preserve: headings, paragraphs, bullet points, numbered lists, tables when possible.
    Do not summarize. Do not explain. Return only the extracted text.
    """.strip()

    response = client.chat.completions.create(
        model=HF_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=OCR_MAX_TOKENS,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""

# ============================================================
# PDF Extraction Pipeline
# ============================================================
def extract_pages(pdf_path: Path, hf_client: InferenceClient) -> List[PageRecord]:
    doc = pymupdf.open(pdf_path)
    records: List[PageRecord] = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        
        # Remove whitespace to accurately check the amount of actual text extracted
        clean_count = len(text.replace(" ", "").replace("\n", ""))

        if clean_count < OCR_TEXT_THRESHOLD:
            try:
                ocr_text = ocr_page(hf_client, page)
                records.append(PageRecord(
                    document_name=pdf_path.name,
                    page_number=page_number,
                    text=ocr_text,
                    extraction_method="huggingface_ocr",
                    ocr_status="SUCCESS" if ocr_text else "EMPTY"
                ))
            except Exception as e:
                records.append(PageRecord(
                    document_name=pdf_path.name,
                    page_number=page_number,
                    text="",
                    extraction_method="huggingface_ocr",
                    ocr_status=f"ERROR: {str(e)}"
                ))
        else:
            records.append(PageRecord(
                document_name=pdf_path.name,
                page_number=page_number,
                text=text,
                extraction_method="pymupdf_text",
                ocr_status="N/A"
            ))

    doc.close()
    return records

def load_and_validate_documents(data_dir: str) -> Dict[str, Any]:
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pdfs = sorted(dir_path.glob("*.pdf"))
    hf_client = create_hf_client()
    
    results = {
        "valid_pages": [],
        "stats": {
            "total_pdfs": len(pdfs),
            "total_pages_processed": 0,
            "total_pymupdf_pages": 0,
            "total_ocr_pages": 0,
            "total_ocr_errors": 0,
            "file_details": []
        }
    }

    for pdf in pdfs:
        file_stats = {
            "file_name": pdf.name,
            "pages": 0,
            "pymupdf_count": 0,
            "ocr_count": 0,
            "characters": 0,
            "status": "Success"
        }
        
        try:
            pages = extract_pages(pdf, hf_client)
            results["valid_pages"].extend(pages)
            
            file_stats["pages"] = len(pages)
            for p in pages:
                file_stats["characters"] += len(p.text)
                if p.extraction_method == "pymupdf_text":
                    file_stats["pymupdf_count"] += 1
                    results["stats"]["total_pymupdf_pages"] += 1
                elif p.extraction_method == "huggingface_ocr":
                    file_stats["ocr_count"] += 1
                    results["stats"]["total_ocr_pages"] += 1
                    if p.ocr_status.startswith("ERROR"):
                        results["stats"]["total_ocr_errors"] += 1
                        
            results["stats"]["total_pages_processed"] += len(pages)
            
        except Exception as e:
            file_stats["status"] = f"Failed to process PDF: {str(e)}"
            
        results["stats"]["file_details"].append(file_stats)

    return results