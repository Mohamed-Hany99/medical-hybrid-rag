import hashlib
import tiktoken
from typing import List, Dict, Any
from ingestion.loader import PageRecord
from ingestion.cleaner import normalize_text, looks_like_heading

# Tokenizer initialization
ENC = tiktoken.get_encoding("cl100k_base")

# Chunking Configuration
MIN_TOKENS = 100
MAX_TOKENS = 700
OVERLAP_TOKENS = 100

def sectionize(page: PageRecord) -> List[Dict[str, str]]:
    """Splits a page into sections based on detected headings."""
    text = normalize_text(page.text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    sections = []
    # Default title if no heading is found at the very beginning
    current_title = page.document_name.rsplit(".", 1)[0]
    buffer = []

    for line in lines:
        if looks_like_heading(line):
            if buffer:
                sections.append({"title": current_title, "text": " ".join(buffer)})
                buffer = []
            current_title = line
        else:
            buffer.append(line)
            
    if buffer:
        sections.append({"title": current_title, "text": " ".join(buffer)})
        
    return sections

def split_by_tokens(text: str, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    """Splits text semantically using token counts."""
    token_ids = ENC.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk = ENC.decode(token_ids[start:end]).strip()
        
        if chunk:
            chunks.append(chunk)
            
        if end >= len(token_ids):
            break
            
        start = max(end - overlap, start + 1)

    # Merge very small tail chunk
    if len(chunks) >= 2:
        last_tokens = len(ENC.encode(chunks[-1]))
        if last_tokens < min_tokens // 2:
            chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
            chunks.pop()

    return chunks

def build_chunks(pages: List[PageRecord]) -> List[Dict[str, Any]]:
    """Converts a list of PageRecords into unified chunks with metadata."""
    chunks = []
    
    for page in pages:
        sections = sectionize(page)
        
        for sec in sections:
            section_title = sec["title"]
            section_text = sec["text"]
            
            text_chunks = split_by_tokens(section_text)
            
            for chunk_index, chunk_text in enumerate(text_chunks):
                # Create a stable, reproducible ID
                raw_id = f"{page.document_name}|{page.page_number}|{section_title}|{chunk_index}|{chunk_text}"
                chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_name": page.document_name,
                    "page_number": page.page_number,
                    "extraction_method": page.extraction_method,
                    "section_title": section_title,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": len(ENC.encode(chunk_text))
                })
                
    return chunks