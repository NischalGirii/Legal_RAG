from src.hybrid_search import detect_query_intent
from src.text_processor import clean_ocr_field


def test_intent_summary_vs_about_vs_principle():
    assert detect_query_intent("निर्णय नं. ९१०० को सारांश दिनुहोस्।") == "CASE_SUMMARY"
    assert detect_query_intent("निर्णय नं. ९०९९ को मुद्दा के सम्बन्धी हो?") == "CASE_ABOUT"
    assert detect_query_intent("यो मुद्दाको मुख्य कानूनी सिद्धान्त के हो?") == "LEGAL_PRINCIPLE"
    assert detect_query_intent("यस मुद्दामा न्यायाधीशहरू को-को हुनुहुन्थ्यो?") == "FACTUAL"
    assert detect_query_intent("निर्णय नं. ९०९९ र ९१०० को तुलना गर्नुहोस्।") == "COMPARISON"
    assert detect_query_intent("दुवै मुद्दाका निवेदकहरू को-को हुन्?") == "COMPARISON"


def test_ocr_field_strips_noise():
    assert "प्रकाश" in clean_ocr_field("प्रप्रकाश वस्ती")
    assert "बैद्यनाथ" in clean_ocr_field("SAT उपाध्याय")
    cleaned = clean_ocr_field("M विद्वान अधिवक्ता भरत जङ्गम")
    assert "भरत" in cleaned
    assert cleaned[0] not in {"M", "U"}
