import re
import unicodedata

NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ASCII_TO_NEPALI = str.maketrans("0123456789", "०१२३४५६७८९")

# ========================================================================
# OCR FIXES (common errors from scanned PDFs) – expanded
# ========================================================================
OCR_FIXES = {
    "SEAT": "बैद्यनाथ",          # Case 9099 judge name
    "का.मु.प्": "",              # Remove stray characters (e.g., from 9100)
    "धानन्यायाधीश": "प्रधानन्यायाधीश",
    "प्रधानन्यायाधीश श्": "प्रधानन्यायाधीश श्री",
    # New fixes based on observed artifacts
    "रीरी": "श्री",
    "काश": "प्रकाश",
    "का.मु.प्, प्रधानन्यायाधीश": "प्रधानन्यायाधीश",
    "प्रधानन्यायाधीश रीरी": "प्रधानन्यायाधीश श्री",
    "प्रधानन्यायाधीश श्, ी": "प्रधानन्यायाधीश श्री",
    "सम्माननीय का.मु.प्, धानन्यायाधीश": "प्रधानन्यायाधीश",
    "सम्माननीय का.मु.प्, प्रधानन्यायाधीश": "प्रधानन्यायाधीश",
    "का.मु.प्": "",              # Remove anyway
    "प्रधानन्यायाधीश श्, ी": "प्रधानन्यायाधीश श्री",  # repeated for safety
    "सम्माननीय का.मु.प्": "",
}

def apply_ocr_fixes(text: str) -> str:
    """Replace known OCR artifacts with correct Nepali text."""
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
    # Apply OCR fixes after basic cleaning
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