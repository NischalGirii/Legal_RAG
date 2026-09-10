import re
import unicodedata

NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ASCII_TO_NEPALI = str.maketrans("0123456789", "०१२३४५६७८९")

OCR_FIXES = {
    "SEAT": "बैद्यनाथ",
    "SAT": "बैद्यनाथ",
    "Seq": "बैद्यनाथ",
    "का.मु.प्": "का.मु. प्रधानन्यायाधीश",
    "धानन्यायाधीश": "प्रधानन्यायाधीश",
    "प्रधानन्यायाधीश श्": "प्रधानन्यायाधीश श्री",
    "रीरी": "श्री",
    "प्रप्रप्रकाश": "प्रकाश",
    "प्रप्रकाश": "प्रकाश",
    "काश": "प्रकाश",
    "प्रधानन्यायाधीश रीरी": "प्रधानन्यायाधीश श्री",
    "प्रधानन्यायाधीश श्, ी": "प्रधानन्यायाधीश श्री",
    "सम्माननीय का.मु.प्, धानन्यायाधीश": "प्रधानन्यायाधीश",
    "सम्माननीय का.मु.प्, प्रधानन्यायाधीश": "प्रधानन्यायाधीश",
    "श्श्री": "श्री",
    "प्रप्रधान": "प्रधान",
    "उत्रेषण": "उत्प्रेषण",
    "मिन्नेको": "मिच्नेको",
    "द्ण्डसजाय": "दण्ड सजाय",
    "अंशबन्डा": "अंशबण्डा",
    "अदालतः": "अदालत:",
}

LEGAL_OCR_PATTERNS = [
    # Common Tesseract corruption: "3g." or "3q." instead of "अ.बं." (अदालती बन्दोबस्त)
    (re.compile(r"(?<!\S)3[gq]\.?\s*(?=[०-९0-9]|१७१|दफा|धारा|\b)", re.I), "अ.बं. "),
    (re.compile(r"(?<!\S)3[gq](?!\S)", re.I), "अ.बं."),
    (re.compile(r"का\.?मु\.?\s*प्(?=धान|[\s,;]|$)", re.I), "का.मु. प्रधानन्यायाधीश "),
    (re.compile(r"(?<!\S)कि\.?\s*नं\.?(?!\S)"), "कि.नं. "),
    (re.compile(r"(?<!\S)पु\.?\s*(?:वे|अ)\.?\s*अदालत(?!\S)"), "पुनरावेदन अदालत "),
    (re.compile(r"(?<!\S)जि\.?\s*अ\.?(?!\S)"), "जिल्ला अदालत "),
    (re.compile(r"(?<!\S)ने\.?\s*का\.?\s*प\.?(?!\S)"), "नेकाप "),
]

_OCR_NOISE_TOKEN = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=,))(?:SAT|SEAT|Seq|Ud|uM|mM|MM|uMM)(?=\s|,|$)",
    re.I,
)
_OCR_LONE_INITIAL = re.compile(r"(?:(?<=\s)|(?<=^))[MuU](?:d)?(?:\s+)(?=विद्वान|अधिवक्ता|सहन्याया|नायब|श्री)")

SPEECH_CORRECTIONS = {
    "मुद्धा": "मुद्दा",
    "मुद्धाहरु": "मुद्दाहरू",
    "मुद्दाहरु": "मुद्दाहरू",
    "प्रहरीष्येवालाई": "प्रहरी सेवालाई",
    "प्रहरीष्येवा": "प्रहरी सेवा",
    "बिसिस्ट": "विशिष्ट",
    "किना": "किन",
    "संदर्विक्ष": "सम्बन्धित छ",
    "न्यादिष्को": "न्यायाधीश को",
    "न्यादिष": "न्यायाधीश",
    "न्यादिश": "न्यायाधीश",
    "तोपेलाई": "तपाईंलाई",
    "जानकारिहा": "जानकारी छ",
    "प्रस्तुत्र": "प्रस्तुत",
    "दिनना": "दिनुहोस्",
}

_SPEECH_CORRECTION_PATTERNS = [
    (re.compile(r"(?<!\S)" + re.escape(typo) + r"(?!\S)"), fix)
    for typo, fix in SPEECH_CORRECTIONS.items()
]


def clean_asr_transcript(text: str) -> str:
    """Corrects Whisper phonetic artifacts, Nepali years, and decision numbers."""
    if not text:
        return ""
    cleaned = text
    for pattern, fix in _SPEECH_CORRECTION_PATTERNS:
        cleaned = pattern.sub(fix, cleaned)

    cleaned = re.sub(r"दु[ईय]?[ा]?जार\s*उन[न|न्]+[ -]?पचास[कोगोमु]*", "२०४९", cleaned, flags=re.I)
    cleaned = re.sub(r"निर्णय\s*(?:नं\.?\s*)?[९9]{2,3}[०0][९9]{2,3}\b", "निर्णय नं. 9099", cleaned)
    cleaned = re.sub(r"निर्णय\s*(?:नं\.?\s*)?[९9]{2,3}[०0][९9][२2]\b", "निर्णय नं. 9099", cleaned)
    cleaned = re.sub(r"निर्णय\s*(?:नं\.?\s*)?[९9][१1][०0][२2]\b", "निर्णय नं. 9100", cleaned)
    cleaned = re.sub(r"नौ[ँं]?\s*[अह]जार\s*(?:एक\s*सय|एक्से[कोगो]?|सय)", "9100", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:एकानब्बे\s*सय|एकानब्बे)\b", "9100", cleaned, flags=re.I)
    cleaned = re.sub(r"नौ[ँं]?\s*[अह]जार\s*(?:उनान्सय|नौ\s*सय)", "9099", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:अजारुनन्सय|उनान्सय)\b", "9099", cleaned, flags=re.I)
    cleaned = re.sub(r"निर्णय\s*(?:नं\.?\s*)?100\b", "निर्णय नं. 9100", cleaned)
    cleaned = re.sub(r"निर्णय\s*(?:नं\.?\s*)?१००\b", "निर्णय नं. 9100", cleaned)

    return cleaned


def clean_text_for_tts(text: str) -> str:
    """Strips Markdown syntax for clean speech output."""
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
    for pattern, replacement in LEGAL_OCR_PATTERNS:
        text = pattern.sub(replacement, text)
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
    text = re.sub(
        r"[^\u0900-\u097F\u0020-\u007Ea-zA-Z0-9\u0964\u0965\t\n\r]",
        " ",
        text,
    )
    text = apply_ocr_fixes(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_digits(text: str) -> str:
    return text.translate(NEPALI_DIGITS) if text else text


def is_valid_devanagari_text(text: str, min_ratio: float = 0.3) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    devanagari = sum("\u0900" <= c <= "\u097F" for c in letters)
    return (devanagari / len(letters)) >= min_ratio


def chunk_text_by_sentences(text: str, max_chars: int = 1000, overlap_sentences: int = 2) -> list[str]:
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


def chunk_by_prakaran(text: str, max_chars: int = 1000, overlap_chars: int = 150) -> list[tuple[str, str]]:
    if not text:
        return []
    pattern = r"(\(?\s*प्रकरण\s*नं\.\s*([०-९0-9]+)\s*\)?)"
    parts = re.split(pattern, text)
    chunks = []
    current_prakaran = None
    current_text = []
    current_len = 0
    for part in parts:
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
    if current_text and current_prakaran is not None:
        chunk_text = " ".join(current_text).strip()
        if chunk_text:
            chunks.append((chunk_text, current_prakaran))
    return chunks