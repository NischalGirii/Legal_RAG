import re
import unicodedata

NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ASCII_TO_NEPALI = str.maketrans("0123456789", "०१२३४५६७८९")

OCR_FIXES = {
    "SEAT": "बैद्यनाथ",
    "का.मु.प्": "",
    "धानन्यायाधीश": "प्रधानन्यायाधीश",
    "प्रधानन्यायाधीश श्": "प्रधानन्यायाधीश श्री",
    "रीरी": "श्री",
    "काश": "प्रकाश",
    "प्रधानन्यायाधीश रीरी": "प्रधानन्यायाधीश श्री",
    "प्रधानन्यायाधीश श्, ी": "प्रधानन्यायाधीश श्री",
    "सम्माननीय का.मु.प्, धानन्यायाधीश": "प्रधानन्यायाधीश",
    "सम्माननीय का.मु.प्, प्रधानन्यायाधीश": "प्रधानन्यायाधीश",
}

def apply_ocr_fixes(text: str) -> str:
    if not text:
        return text
    for wrong, correct in OCR_FIXES.items():
        text = text.replace(wrong, correct)
    return text

def clean_devanagari_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = apply_ocr_fixes(text)
    return text.strip()

def normalize_digits(text: str) -> str:
    return text.translate(NEPALI_DIGITS) if text else text

def is_valid_devanagari_text(text: str, min_ratio: float = 0.4) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    devanagari = sum("\u0900" <= c <= "\u097F" for c in letters)
    return (devanagari / len(letters)) >= min_ratio

def chunk_text_by_sentences(text: str, max_chars: int = 500, overlap_sentences: int = 1) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[।!?])\s+|\n+", text) if s.strip()]
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunks.append(" ".join(current).strip())
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = overlap + [sentence]
            current_len = sum(len(x) + 1 for x in current) - 1
        else:
            current.append(sentence)
            current_len += add_len
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

def char_ngram_tokenize(text: str, n: int = 3) -> list[str]:
    text = re.sub(r"\s+", "", text or "").lower()
    if len(text) <= n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def clean_and_repair_nepali_output(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# ========================================================================
# NEW: Prakaran-aware chunking for NKP PDFs
# ========================================================================
def chunk_by_prakaran(text: str, max_chars: int = 600, overlap_chars: int = 100) -> list[tuple[str, str]]:
    """
    Split text by प्रकरण नं. markers and return (chunk_text, prakaran_no).
    """
    if not text:
        return []
    # Regex to find "प्रकरण नं. X" or "(प्रकरण नं. X)"
    pattern = r"(\(?\s*प्रकरण\s*नं\.\s*([०-९0-9]+)\s*\)?)"
    parts = re.split(pattern, text)
    
    chunks = []
    current_prakaran = None
    current_text = []
    current_len = 0
    
    for i, part in enumerate(parts):
        # If part matches the pattern, it's a paragraph marker
        if re.match(pattern, part, re.I):
            # If we have accumulated text, save it with the previous paragraph number
            if current_text and current_prakaran is not None:
                chunk_text = " ".join(current_text).strip()
                if chunk_text:
                    chunks.append((chunk_text, current_prakaran))
            # Start a new paragraph
            num_match = re.search(r"([०-९0-9]+)", part)
            current_prakaran = num_match.group(1) if num_match else None
            current_text = []
            current_len = 0
        else:
            # Add text to current paragraph
            if part.strip():
                sentences = re.split(r"(?<=[।!?])\s+", part)
                for sent in sentences:
                    if not sent.strip():
                        continue
                    if current_len + len(sent) > max_chars and current_text:
                        chunk_text = " ".join(current_text).strip()
                        if chunk_text:
                            chunks.append((chunk_text, current_prakaran))
                        overlap = current_text[-overlap_chars:] if overlap_chars > 0 else []
                        current_text = overlap + [sent]
                        current_len = sum(len(x) for x in current_text)
                    else:
                        current_text.append(sent)
                        current_len += len(sent)
    
    # Add the last chunk
    if current_text and current_prakaran is not None:
        chunk_text = " ".join(current_text).strip()
        if chunk_text:
            chunks.append((chunk_text, current_prakaran))
    
    return chunks