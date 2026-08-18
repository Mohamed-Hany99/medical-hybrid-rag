# Day 1 — Document Ingestion / Retrieval

This project implements the Day-1 pipeline from the supplied presentation/notebook for the uploaded cardiovascular-health PDFs.

## Pipeline

PDFs → parsing → cleaning → repeated header/footer removal → section detection → section-aware 400–800 token chunks → metadata → embeddings → ChromaDB → semantic retrieval.

A PDF page with little/no extractable text automatically falls back to OCR using Tesseract.

## Project structure

- `data/` — the 7 supplied PDFs
- `audit.py` — quick PDF audit
- `ingest.py` — complete ingestion + embedding + ChromaDB indexing
- `retrieve.py` — semantic search against the index
- `config.py` — paths, chunk sizes, model and settings
- `requirements.txt` — dependencies
- `chroma_db/` — created after indexing

## Windows setup

1. Install Python 3.11+.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install packages:

```powershell
pip install -r requirements.txt
```

4. Run the audit:

```powershell
python audit.py
```

5. Install Tesseract OCR on Windows and make sure `tesseract.exe` is in PATH if the scanned `CVDS.pdf` needs OCR.

6. Build the index:

```powershell
python ingest.py
```

7. Test retrieval:

```powershell
python retrieve.py "What are the recommendations for physical activity?"
python retrieve.py "How does smoking affect cardiovascular health?"
python retrieve.py "What are the components of Life's Essential 8?"
python retrieve.py "How can diet help with cardiovascular disease prevention?"
python retrieve.py "What is the role of cholesterol in cardiovascular health?"
```

## Notes

- `all-MiniLM-L6-v2` is used as a local embedding model to avoid requiring an API key.
- `source_url` is intentionally left empty unless official URLs are supplied; the document filename/page/section remain available for traceability.
- The main strategy is section-aware chunking. The code keeps a fixed-size token splitter as the underlying overflow mechanism for long sections.
