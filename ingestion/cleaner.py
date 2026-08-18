import re

def normalize_text(text: str) -> str:
    """Normalize extracted/OCR text while preserving useful paragraph boundaries."""
    if not text:
        return ""

    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    
    # Repair hyphenation caused by line wrapping
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    
    # Normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    
    # Preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Turn single line breaks into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text.strip()

def looks_like_heading(line: str) -> bool:
    """Detect if a line is a section heading."""
    s = line.strip()

    if not s or len(s) > 140 or len(s.split()) > 18:
        return False

    if re.match(r"^(figure|table|source|references?)\b", s, re.I):
        return False

    if re.match(r"^\d+[.)]\s+", s) or re.match(r"^[IVX]+[.)]\s+", s):
        return True

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s)
    if len(words) >= 2:
        titleish = sum(word[0].isupper() for word in words) / len(words) >= 0.65
        all_caps = (s.upper() == s and any(char.isalpha() for char in s))
        
        if titleish or all_caps:
            return True

    return False