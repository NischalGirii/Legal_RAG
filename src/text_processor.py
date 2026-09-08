import re
import unicodedata

NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ASCII_TO_NEPALI = str.maketrans("0123456789", "०१२३४५६७८९")

# Mapping for spoken Devanagari numbers to canonical 4‑digit forms
DEVANAGARI_NUM_WORDS = {
    "उन्नाइस": "19",
    "उन्नाइससय": "1900",
    "उन्नाइस सय": "1900",
    "उन्नाइस सय तीन": "1903",
    "एकानब्बे": "91",
    "एकानब्बेसय": "9100",
    "एक सय": "100",
    "नौ हजार": "9000",
    "नौहजार": "9000",
    "उनन्चालीस": "49",
    # add more as needed
}

OCR_FIXES = {
    "SEAT": "बैद्यनाथ",
    "SAT": "बैद्यनाथ",
    "Seq": "बैद्यनाथ",
    "का.मु.प्": "",
    "धानन्यायाधीश": "प्रधानन्यायाधीश",
    "प्रधानन्यायाधीश श्": "प्रधानन्यायाधीश श्री",
    "रीरी": "श्री",
    "प्रप्रप्रकाश": "प्रकाश",
    "प्रप्रकाश": "प्रकाश",
    "प्रधानन्यायाधीश रीरी": "प्रधानन्यायाधीश श्री",
    "प्रधानन्यायाधीश श्, ी": "प्रधानन्यायाधीश श्री",
    "सम्माननीय का.मु.प्, धानन्यायाधीश": "प्रधानन्यायाधीश",
    "सम्माननीय का.मु.प्, प्रधानन्यायाधीश": "प्रधानन्यायाधीश",
    "श्श्री": "श्री",
    "प्रप्रधान": "प्रधान",
    "उत्रेषण": "उत्प्रेषण",
    "पुनरावलोकन": "पुनरावलोकन",
    "सर्वोच्चअदालत": "सर्वोच्च अदालत",
    "विद्वानअधिवक्ता": "विद्वान अधिवक्ता",
    "सहन्यायाधिवक्ता": "सहन्यायाधिवक्ता",
    "नायबमहान्यायाधिवक्ता": "नायब महान्यायाधिवक्ता",
}

_OCR_NOISE_TOKEN = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=,))(?:SAT|SEAT|Seq|Ud|uM|mM|MM|uMM|Yel|Yad|YUASA)(?=\s|,|$|:)",
    re.I,
)
_OCR_LONE_INITIAL = re.compile(r"(?:(?<=\s)|(?<=^))[MuU](?:d)?(?:\s+)(?=विद्वान|अधिवक्ता|सहन्याया|नायब|श्री|काठमाडौं|नेपाल|फैसला)")

SPEECH_CORRECTIONS = {
    "मुद्धा": "मुद्दा",
    "मुद्धाहरु": "मुद्दाहरू",
    "मुद्दाहरु": "मुद्दाहरू",
    "निन्याय": "निर्णय",
    "निर्णया": "निर्णय",
    "नियायाधीश": "न्यायाधीश",
    "न्यादिष": "न्यायाधीश",
    "न्यादिश": "न्यायाधीश",
    "न्यादिष्को": "न्यायाधीशको",
    "न्यायाधिश": "न्यायाधीश",
    "प्रहरीष्येवालाई": "प्रहरी सेवालाई",
    "प्रहरीष्येवा": "प्रहरी सेवा",
    "बिसिस्ट": "विशिष्ट",
    "किना": "किन",
    "संदर्विक्ष": "सम्बन्धित छ",
    "तोपेलाई": "तपाईंलाई",
    "तपई": "तपाईं",
    "तोपाई": "तपाईं",
    "तबैचना": "तपाईंसँग",
    "समा": "सँग",
    "ग्यान": "ज्ञान",
    "जानकारिहा": "जानकारी छ",
    "कुनकुर": "कुन-कुन",
    "कुनकुन": "कुन-कुन",
    "केके": "के-के",
    "मुद्दाच्छ": "मुद्दा छन्",
    "अरुछन्": "अरू छन्",
    "कोको": "को-को",
    "प्रस्तुत्र": "प्रस्तुत",
    "दिनना": "दिनुहोस्",
}

def clean_asr_transcript(text: str) -> str:
    """Corrects Whisper phonetic artifacts, Nepali years, and spoken decision numbers without corrupting valid digits."""
    if not text:
        return ""
    cleaned = text

    # Word-level speech corrections
    for typo, fix in SPEECH_CORRECTIONS.items():
        cleaned = re.sub(rf"(?<!\S){re.escape(typo)}(?!\S)", fix, cleaned)
        cleaned = cleaned.replace(typo, fix)

    # 1. Police Regulations year: "दुयाजार उनन पचास" / "दुई हजार उनन्पचास" -> २०४९
    cleaned = re.sub(r"दु[ईय]?[ा]?जार\s*उन[न|न्]+[ -]?पचास[कोगोमु]*", "२०४९", cleaned, flags=re.I)

    # 2. Fix spoken 9099 variations ("नौ हजार उनान्सय", "नौ हजार नौ सय नौ")
    cleaned = re.sub(r"नौ[ँं]?\s*[अह]जार\s*(?:उनान्सय|नौ\s*सय\s*(?:उनान्सय|नौ))", "9099", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:अजारउनान्सय|उनान्सय|नौहजारउनान्सय)\b", "9099", cleaned, flags=re.I)

    # 3. Fix spoken 9100 variations ("नौ हजार एक सय")
    cleaned = re.sub(r"नौ[ँं]?\s*[अह]जार\s*(?:एक\s*सय|एक्से[कोगो]?)", "9100", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:एकानब्बे\s*सय|एकानब्बेसय)\b", "9100", cleaned, flags=re.I)

    # 4. Standardize explicit decision references with spaces
    cleaned = re.sub(r"निर्णय\s*न[म्ं\.]*\s*([०-९0-9]+)", r"निर्णय नं. \1", cleaned)

    return cleaned

def clean_text_for_tts(text: str) -> str:
    """Strips Markdown syntax (asterisks, bullet points, headers) for clean speech."""
    if not text:
        return ""
    s = re.sub(r"#+\s*", "", text)
    s = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)
    s = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", s)
    s = re.sub(r"^\s*[-*•]\s*", "", s, flags=re.M)
    s = re.sub(r"\n+", "। ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def apply_ocr_fixes(text: str) -> str:
    if not text:
        return text
    for wrong, correct in OCR_FIXES.items():
        text = text.replace(wrong, correct)
    return text

def clean_ocr_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = [clean_ocr_field(v) for v in value.values() if v]
        return ", ".join(p for p in parts if p)
    if isinstance(value, (list, tuple)):
        return ", ".join(clean_ocr_field(v) for v in value if v)
    text = apply_ocr_fixes(str(value)).strip()
    if not text or text.upper() in {"UNKNOWN", "N/A", "NONE"}:
        return ""
    text = _OCR_NOISE_TOKEN.sub(" ", text)
    text = _OCR_LONE_INITIAL.sub(" ", text)
    text = re.sub(r"^[\s,;:।|MUu/-]+", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ", ", text)
    return text.strip(" ,;|-")

def clean_devanagari_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", " ", text)
    # Retain Devanagari range, standard ASCII, punctuation, danda (\u0964, \u0965)
    text = re.sub(
        r"[^\u0900-\u097F\u0020-\u007Ea-zA-Z0-9\u0964\u0965\u200C\u200D\t\n\r]",
        " ",
        text,
    )
    text = apply_ocr_fixes(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_digits(text: str) -> str:
    return text.translate(NEPALI_DIGITS) if text else text

def to_nepali_digits(text: str) -> str:
    return text.translate(ASCII_TO_NEPALI) if text else text

def is_valid_devanagari_text(text: str, min_ratio: float = 0.4) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    devanagari = sum("\u0900" <= c <= "\u097F" for c in letters)
    return (devanagari / len(letters)) >= min_ratio

def chunk_text_by_sentences(text: str, max_chars: int = 1200, overlap_sentences: int = 2) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[।!?])\s+|\n{2,}", text) if s.strip()]
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
    """Generates both word tokens and character n-grams for robust Devanagari lexical retrieval."""
    if not text:
        return []
    raw = clean_devanagari_text(text)
    words = [w.strip() for w in re.split(r"[\s,।!?\":;()\[\]{}]+", raw) if len(w.strip()) > 1]
    
    # Generate character n-grams from words
    ngrams = []
    for w in words:
        if len(w) <= n:
            ngrams.append(w)
        else:
            for i in range(len(w) - n + 1):
                ngrams.append(w[i:i+n])
    
    return words + ngrams

def clean_and_repair_nepali_output(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def chunk_by_prakaran(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[tuple[str, str]]:
    """Chunks text while preserving prakaran (paragraph number) boundaries and context."""
    if not text:
        return []
    pattern = r"(\(?\s*प्रकरण\s*नं\.?\s*([०-९0-9]+)\s*\)?)"
    parts = re.split(pattern, text)
    chunks = []
    current_prakaran = None
    current_text = []
    current_len = 0
    
    for i, part in enumerate(parts):
        if not part:
            continue
        if re.match(pattern, part, re.I):
            if current_text and current_prakaran is not None:
                chunk_text = " ".join(current_text).strip()
                if chunk_text:
                    chunks.append((chunk_text, current_prakaran))
            num_match = re.search(r"([०-९0-9]+)", part)
            current_prakaran = num_match.group(1) if num_match else None
            current_text = []
            current_len = 0
        else:
            if part.strip():
                sentences = re.split(r"(?<=[।!?])\s+|\n+", part)
                for sent in sentences:
                    if not sent.strip():
                        continue
                    if current_len + len(sent) > max_chars and current_text:
                        chunk_text = " ".join(current_text).strip()
                        if chunk_text:
                            chunks.append((chunk_text, current_prakaran))
                        overlap = current_text[-1:] if current_text else []
                        current_text = overlap + [sent]
                        current_len = sum(len(x) + 1 for x in current_text)
                    else:
                        current_text.append(sent)
                        current_len += len(sent) + 1
                        
    if current_text:
        chunk_text = " ".join(current_text).strip()
        if chunk_text:
            chunks.append((chunk_text, current_prakaran))
            
    return chunks

# ----- New functions for number preservation -----
def preserve_original_decision_number(text: str) -> tuple[str, str]:
    """
    Extracts decision number from text, returns (original_devanagari, normalized_english).
    Example: "निर्णय नं. १९०३" -> ("१९०३", "1903")
    """
    match = re.search(r"निर्णय\s*नं\.?\s*([०-९]+)", text)
    if match:
        dev = match.group(1)
        eng = dev.translate(NEPALI_DIGITS)
        return dev, eng
    return "", ""

def normalize_spoken_decision(text: str) -> str | None:
    """Convert spoken decision number variants to 4-digit canonical form."""
    for phrase, num in DEVANAGARI_NUM_WORDS.items():
        if phrase in text:
            return num
    return None