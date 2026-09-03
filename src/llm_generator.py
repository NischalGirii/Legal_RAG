import os
import re
import time
import json
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv
from src.text_processor import (
    clean_and_repair_nepali_output,
    normalize_digits,
    apply_ocr_fixes,
    clean_ocr_field,
)
from src.config import CASE_SUMMARIES_PATH, LLM_MODEL
from src.hybrid_search import detect_query_intent, chunks_for_decision, decision_exists

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_groq_client = None
DEFAULT_MODEL = LLM_MODEL

MANUAL_SUMMARIES = {}
if os.path.exists(CASE_SUMMARIES_PATH):
    try:
        with open(CASE_SUMMARIES_PATH, "r", encoding="utf-8") as f:
            MANUAL_SUMMARIES = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load manual summaries: {e}")


def get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

NO_INFO = "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
GROQ_UNAVAILABLE = "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"
SERVER_ERROR = "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"

# ========================================================================
# DETERMINISTIC ANSWERS for common legal terms
# ========================================================================
def get_deterministic_answer(query: str) -> str | None:
    """
    Return a hardcoded answer for frequently asked questions,
    bypassing retrieval and LLM for speed and reliability.
    """
    q = query.lower()
    q = re.sub(r"[^\w\s\u0900-\u097f]", "", q)  # keep only words and Devanagari

    # उत्प्रेषण
    if "उत्प्रेषण" in q and ("के हो" in q or "बारे" in q):
        return "उत्प्रेषण (Certiorari) भनेको तल्लो अदालत वा निकायको आदेश, निर्णय वा कार्यको पुनरावलोकन गर्न माग गरिएको रिट हो। यसले अधिकारक्षेत्र बाहिर गएको वा कानूनी त्रुटि भएको कार्यलाई रद्द गर्न माग गरिन्छ।"
    if "परमादेश" in q and ("के हो" in q or "बारे" in q):
        return "परमादेश (Mandamus) भनेको कुनै सार्वजनिक निकायलाई आफ्नो कर्तव्य पालना गर्न आदेश दिन माग गरिएको रिट हो। यसले निकायलाई आफ्नो वैधानिक दायित्व पूरा गर्न बाध्य पार्छ।"
    if "बन्दीप्रत्यक्षीकरण" in q and ("के हो" in q or "बारे" in q):
        return "बन्दीप्रत्यक्षीकरण (Habeas Corpus) भनेको कुनै व्यक्तिलाई अवैध रूपमा हिरासतमा राखिएको भने उसलाई अदालत समक्ष पेश गर्न माग गरिएको रिट हो। यसले व्यक्तिको स्वतन्त्रताको हकको संरक्षण गर्दछ।"
    if "न्यायिक पुनरावलोकन" in q and ("के हो" in q or "बारे" in q):
        return "न्यायिक पुनरावलोकन (Judicial Review) भनेको अदालतले विधायिका वा कार्यपालिकाका कार्यहरू संविधानसँग मिल्दो छ कि छैन भनी जाँच गर्ने अधिकार हो।"
    if "रिट" in q and ("के हो" in q or "बारे" in q):
        return "रिट (Writ) भनेको अदालतबाट जारी गरिने एक लिखित आदेश हो, जसले कुनै व्यक्ति वा निकायलाई कानूनी कर्तव्य पालना गर्न, अधिकार संरक्षण गर्न, वा अवैध कार्य रोक्न निर्देशन दिन्छ। नेपालको संविधानले उत्प्रेषण, परमादेश, बन्दीप्रत्यक्षीकरण, प्रतिषेध, र अधिकारपृच्छा गरी पाँच प्रकारका रिटको व्यवस्था गरेको छ।"
    return None

def truncate_at_sentence(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    search_start = max(0, max_len - 150)
    last_punct = -1
    for punct in ["।", "\n", "?", "!"]:
        pos = truncated.rfind(punct, search_start)
        if pos > last_punct:
            last_punct = pos
    if last_punct != -1:
        return truncated[:last_punct + 1]
    last_space = truncated.rfind(" ", search_start)
    if last_space != -1:
        return truncated[:last_space] + " ..."
    return truncated + " ..."

# ========================================================================
# CASE CARDS + COMPARISON
# ========================================================================
def _party_from_manual(manual: dict, kind: str) -> str:
    parties = manual.get("parties") or ""
    if kind == "petitioner":
        m = re.search(r"(?:निवेदक|पुनरावेदक)\s*[:：]\s*([^\n]+)", parties)
        return clean_ocr_field(m.group(1)) if m else clean_ocr_field(parties)
    m = re.search(r"(?:विपक्षी|प्रत्यर्थी)\s*[:：]\s*([^\n]+)", parties)
    return clean_ocr_field(m.group(1)) if m else ""


def _judges_from_manual(manual: dict) -> str:
    extracted = _extract_from_manual(manual, "judges")
    if extracted:
        return extracted.replace("न्यायाधीश:", "").strip()
    return clean_ocr_field(manual.get("judges", ""))


def build_case_card(dec_no: str, metadata_info: dict, chunk_metadata: list) -> dict:
    dec_no = normalize_digits(str(dec_no))
    case_id = f"decision_{dec_no}"
    meta_cases = (metadata_info or {}).get("case_metadata") or {}
    ingest = meta_cases.get(case_id) or {}
    manual = MANUAL_SUMMARIES.get(case_id) or {}
    chunks = chunks_for_decision(chunk_metadata, dec_no, case_id)
    header = next((c for c in chunks if c.get("is_header")), chunks[0] if chunks else {})

    subject = clean_ocr_field(
        manual.get("subject") or ingest.get("subject") or header.get("subject") or ""
    )
    if not subject and manual.get("introduction"):
        m = re.search(r"मुद्दाको प्रकार\s*[:：]\s*([^\n।]+)", manual["introduction"])
        if m:
            subject = clean_ocr_field(m.group(1))

    card = {
        "decision_no": dec_no,
        "case_id": case_id,
        "known": bool(chunks or manual or ingest),
        "from_manual": bool(manual),
        "source": ingest.get("source") or header.get("source") or "",
        "date": clean_ocr_field(ingest.get("date") or header.get("date") or ""),
        "court": clean_ocr_field(ingest.get("court") or header.get("court") or "सर्वोच्च अदालत"),
        "subject": subject,
        "judges": _judges_from_manual(manual) if manual else clean_ocr_field(header.get("judges") or ingest.get("judges")),
        "petitioner": _party_from_manual(manual, "petitioner") if manual else clean_ocr_field(
            (ingest.get("parties") or {}).get("appellant") if isinstance(ingest.get("parties"), dict) else header.get("parties")
        ),
        "respondent": _party_from_manual(manual, "respondent") if manual else clean_ocr_field(
            (ingest.get("parties") or {}).get("respondent") if isinstance(ingest.get("parties"), dict) else ""
        ),
        "lawyers": clean_ocr_field(manual.get("lawyers") or ""),
        "appellant_lawyer": clean_ocr_field(manual.get("lawyers") or ingest.get("appellant_lawyer") or header.get("appellant_lawyer")),
        "respondent_lawyer": clean_ocr_field(ingest.get("respondent_lawyer") or header.get("respondent_lawyer")),
        "provisions": clean_ocr_field(ingest.get("provisions") or header.get("provisions")),
        "key_facts": manual.get("key_facts") or "",
        "legal_questions": manual.get("legal_questions") or "",
        "reasoning": manual.get("reasoning") or "",
        "final_order": manual.get("final_order") or clean_ocr_field(ingest.get("final_order") or header.get("final_order")),
        "legal_principle": manual.get("legal_principle") or clean_ocr_field(header.get("legal_principle")),
        "introduction": manual.get("introduction") or "",
    }
    if not card["lawyers"]:
        parts = [p for p in (card["appellant_lawyer"], card["respondent_lawyer"]) if p]
        card["lawyers"] = " / ".join(parts)
    return card


def _card_context(card: dict) -> str:
    if not card.get("known"):
        return f"निर्णय नं. {card['decision_no']} – यस ज्ञानकोषमा अनुक्रमित छैन।"
    lines = [
        f"निर्णय नं.: {card['decision_no']}",
        f"स्रोत PDF: {card['source'] or 'उपलब्ध छैन'}",
        f"विषय: {card['subject'] or 'उपलब्ध छैन'}",
        f"मिति: {card['date'] or 'उपलब्ध छैन'}",
        f"अदालत: {card['court']}",
        f"न्यायाधीश: {card['judges'] or 'उपलब्ध छैन'}",
        f"निवेदक: {card['petitioner'] or 'उपलब्ध छैन'}",
        f"विपक्षी: {card['respondent'] or 'उपलब्ध छैन'}",
        f"कानून व्यवसायी: {card['lawyers'] or 'उपलब्ध छैन'}",
        f"प्रावधान: {card['provisions'] or 'उपलब्ध छैन'}",
        f"मुख्य तथ्य: {card['key_facts'] or 'उपलब्ध छैन'}",
        f"कानूनी प्रश्न: {card['legal_questions'] or 'उपलब्ध छैन'}",
        f"निष्कर्ष: {card['reasoning'] or 'उपलब्ध छैन'}",
        f"अन्तिम आदेश: {card['final_order'] or 'उपलब्ध छैन'}",
        f"कानूनी सिद्धान्त: {card['legal_principle'] or 'उपलब्ध छैन'}",
    ]
    return "\n".join(lines)


def _format_structured_comparison(cards: list[dict]) -> str:
    blocks = []
    for card in cards:
        if not card["known"]:
            blocks.append(f"### निर्णय नं. {card['decision_no']}\nयस ज्ञानकोषमा यो निर्णय अनुक्रमित छैन।")
            continue
        blocks.append(
            f"### निर्णय नं. {card['decision_no']}\n"
            f"* **विषय:** {card['subject'] or 'उपलब्ध छैन'}\n"
            f"* **मिति:** {card['date'] or 'उपलब्ध छैन'}\n"
            f"* **निवेदक:** {card['petitioner'] or 'उपलब्ध छैन'}\n"
            f"* **विपक्षी:** {card['respondent'] or 'उपलब्ध छैन'}\n"
            f"* **न्यायाधीश:** {card['judges'] or 'उपलब्ध छैन'}\n"
            f"* **मुख्य तथ्य:** {truncate_at_sentence(card['key_facts'], 420) if card['key_facts'] else 'उपलब्ध छैन'}\n"
            f"* **अदालतको निष्कर्ष / अन्तिम आदेश:** {truncate_at_sentence(card['final_order'], 360) if card['final_order'] else 'उपलब्ध छैन'}\n"
            f"* **मुख्य कानूनी सिद्धान्त:** {card['legal_principle'] or 'उपलब्ध छैन'}"
        )
    if len(cards) >= 2 and all(c["known"] for c in cards):
        rows = [
            "| पक्ष | " + " | ".join(c["decision_no"] for c in cards) + " |",
            "| --- | " + " | ".join("---" for _ in cards) + " |",
        ]
        for label, key in [
            ("विषय", "subject"),
            ("निवेदक", "petitioner"),
            ("अन्तिम आदेश", "final_order"),
        ]:
            cells = []
            for c in cards:
                val = (c.get(key) or "—").replace("\n", " ")
                cells.append(truncate_at_sentence(val, 80))
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
        blocks.append("### मुख्य फरक\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def generate_comparison_answer(query: str, decision_numbers: list, metadata_info: dict,
                               chunk_metadata: list, collection, model, bm25,
                               top_k=5, alpha=0.15) -> str | None:
    if not decision_numbers or len(decision_numbers) < 2:
        return None

    unique_numbers = list(dict.fromkeys(normalize_digits(str(n)) for n in decision_numbers))
    cards = [build_case_card(n, metadata_info, chunk_metadata) for n in unique_numbers]
    q_lower = (query or "").lower()

    if any(kw in q_lower for kw in ["निवेदक", "पुनरावेदक", "पक्षकार"]) and "विपक्षी" not in q_lower:
        lines = []
        for card in cards:
            name = card["petitioner"] or "उपलब्ध जानकारीमा निवेदक स्पष्ट छैन।"
            lines.append(f"**निर्णय नं. {card['decision_no']}:** {name}")
        return "\n".join(lines)

    if all(card.get("from_manual") for card in cards):
        return _format_structured_comparison(cards)

    contexts = "\n\n".join(f"--- Case {c['decision_no']} ---\n{_card_context(c)}" for c in cards)
    system_prompt = f"""तपाईं नेपाली कानूनी सहायक हुनुहुन्छ। तलका Case Card बाहेकको तथ्य नलेख्नुहोस्।
OCR टुक्रा (M, Ud, SAT, प्रप्रकाश) लाई उत्तरमा नदेखाउनुहोस्; नाम सफा रूपमा लेख्नुहोस्।
प्रत्येक निर्णयलाई छुट्टै राख्नुहोस् र अन्त्यमा मुख्य फरक दिनुहोस्।
यदि कुनै निर्णय ज्ञानकोषमा छैन भने त्यही भन्नुहोस्।

{contexts}"""

    if not get_groq_client():
        return _format_structured_comparison(cards)

    response = _call_groq(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": query.strip()}],
        DEFAULT_MODEL,
        1200,
        stream=False,
    )
    if response is None:
        return _format_structured_comparison(cards)
    raw_answer = response.choices[0].message.content
    if raw_answer and raw_answer.strip():
        return apply_ocr_fixes(clean_and_repair_nepali_output(raw_answer))
    return _format_structured_comparison(cards)


# ========================================================================
# FACTUAL EXTRACTION (improved with content-based fallback)
# ========================================================================

def _extract_from_manual(manual: dict, category: str) -> str | None:
    parties = manual.get("parties", "")
    intro = manual.get("introduction", "")

    if category == "judges":
        judges = clean_ocr_field(manual.get("judges", ""))
        if judges:
            return f"न्यायाधीश: {judges}"
        bench = re.search(
            r"((?:सम्माननीय|माननीय|प्रधानन्यायाधीश|न्यायाधीश).{8,220}इजलास)",
            intro,
        )
        if bench:
            return f"न्यायाधीश: {clean_ocr_field(bench.group(1))}"
        judge_match = re.search(r"(?:न्यायाधीशहरू|न्यायाधीश)\s*[:：]?\s*([^\n।]+)", intro)
        if judge_match:
            raw = clean_ocr_field(judge_match.group(1))
            raw = re.sub(r"(सम्माननीय|माननीय)\s*", "", raw)
            if raw.strip():
                return f"न्यायाधीश: {raw.strip()}"
        return None
    if category == "petitioner":
        if "निवेदक:" in parties:
            start = parties.find("निवेदक:") + len("निवेदक:")
            end = parties.find("विपक्षी:", start)
            val = parties[start: end if end != -1 else len(parties)].strip()
            if val:
                return f"निवेदक: {val}"
        m = re.search(r"(?:निवेदक|पुनरावेदक)\s*[:：]\s*([^\n]+)", parties or intro)
        if m:
            val = m.group(1).strip()
            if val:
                return f"निवेदक/पुनरावेदक: {val}"
        return None
    if category == "respondent":
        if "विपक्षी:" in parties:
            start = parties.find("विपक्षी:") + len("विपक्षी:")
            val = parties[start:].split("\n")[0].strip()
            if val:
                return f"विपक्षी: {val}"
        m = re.search(r"(?:विपक्षी|प्रत्यर्थी)\s*[:：]\s*([^\n]+)", parties or intro)
        if m:
            val = m.group(1).strip()
            if val:
                return f"विपक्षी: {val}"
        return None
    if category == "lawyers":
        lawyers = manual.get("lawyers", "")
        if lawyers and lawyers.strip():
            return lawyers.strip()
        return None
    if category == "date":
        m = re.search(r"फैसला\s*मिति\s*[:：]\s*([0-9/.\-]+)", intro)
        if m:
            val = m.group(1).strip()
            if val:
                return f"फैसला मिति: {val}"
        return None
    if category == "final_order":
        order = manual.get("final_order", "")
        if order and order.strip():
            return order.strip()
        return None
    if category == "case_number":
        m = re.search(r"निर्णय\s*नं\.?\s*[:：]?\s*([0-9]+)", intro)
        if m:
            val = m.group(1).strip()
            if val:
                return f"निर्णय नं.: {val}"
        return None
    if category == "subject":
        subject = manual.get("subject", "") or manual.get("case_type", "")
        if subject and subject.strip():
            return subject.strip()
        return None
    if category == "provisions":
        provisions = manual.get("provisions", "")
        if provisions and provisions.strip():
            return provisions.strip()
        return None
    return None

def answer_factual_query(query: str, retrieved_items: list, current_case: dict = None) -> str | None:
    """
    Try to answer factual questions using metadata or header chunk.
    If metadata is empty, scan the content of all retrieved chunks.
    Returns None if answer is not found or empty, so LLM can be invoked.
    """
    if not retrieved_items:
        return None

    q_lower = query.lower()

    # Negative/hallucination queries
    if any(kw in q_lower for kw in ["नभएको", "उल्लेख नभएको", "छैन", "mentioned होइन"]):
        return "उपलब्ध निर्णय Context मा उल्लेख नभएको जानकारी निश्चित रूपमा पहिचान गर्न पर्याप्त आधार छैन। त्यसैले अनुमान गरेर उत्तर दिन मिल्दैन।"

    header_item = next((item for item in retrieved_items if item.get("is_header")), None)
    all_content = "\n".join([item.get("content", "") for item in retrieved_items])

    # ---- Try metadata first ----
    if header_item:
        if any(kw in q_lower for kw in ["अन्तिम आदेश", "निष्कर्ष", "सदर", "उल्टी", "खारेज"]):
            val = clean_ocr_field(header_item.get("final_order"))
            if val and val != "UNKNOWN":
                return f"अन्तिम आदेश: {val}"
        if any(kw in q_lower for kw in ["सिद्धान्त", "प्रतिपादन", "precedent"]):
            val = clean_ocr_field(header_item.get("legal_principle"))
            if val and val != "UNKNOWN":
                return f"मुख्य कानूनी सिद्धान्त: {val}"
        if any(kw in q_lower for kw in ["नजिर", "पूर्व निर्णय", "अघिल्ला"]):
            val = clean_ocr_field(header_item.get("precedents"))
            if val and val != "UNKNOWN":
                return f"अघिल्ला नजिरहरू: {val}"

    # ---- Detect category with stricter conditions ----
    category = None

    if any(kw in q_lower for kw in ["कानून व्यवसायी", "अधिवक्ता", "वकील", "बहस गर्ने"]):
        category = "lawyers"
    elif (re.search(r"\b(कसले|को-को|को हुन्|नाम|पक्षकार)\b", q_lower) and
          ("निवेदक" in q_lower or "पुनरावेदक" in q_lower)):
        category = "petitioner"
    elif (re.search(r"\b(कसले|को-को|को हुन्|नाम|पक्षकार)\b", q_lower) and
          ("विपक्षी" in q_lower or "प्रत्यर्थी" in q_lower)):
        category = "respondent"
    elif any(kw in q_lower for kw in ["न्यायाधीश", "इजलास", "बेन्च", "हेर्नुभएको"]):
        category = "judges"
    elif "विषय" in q_lower or "प्रकार" in q_lower:
        category = "subject"
    elif "मिति" in q_lower or "फैसला मिति" in q_lower or "कहिले" in q_lower:
        category = "date"
    elif ("अन्तिम" in q_lower and "आदेश" in q_lower) or "निष्कर्ष" in q_lower or ("फैसला" in q_lower and any(w in q_lower for w in ["के", "कस्तो", "गरेको"])):
        category = "final_order"
    elif "मुद्दा" in q_lower and "नं" in q_lower:
        category = "case_number"
    elif ("धारा" in q_lower or "दफा" in q_lower) and not re.search(r"(?:धारा|दफा)\s*[०-९0-9]+", query):
        category = "provisions"

    if category is None:
        return None

    # ---- Try manual summary ----
    case_id = current_case.get("case_id") if current_case else None
    manual = MANUAL_SUMMARIES.get(case_id) if case_id else None
    if manual:
        ans = _extract_from_manual(manual, category)
        if ans and ans.strip():
            return ans

    # ---- Fallback: header metadata ----
    if header_item:
        if category == "judges":
            judges = header_item.get("judges", "")
            if judges and judges.strip():
                return f"न्यायाधीश: {clean_ocr_field(judges)}"
            match = re.search(r"न्यायाधीश\s*[:：]\s*([^\n]+)", header_item.get("content", ""))
            if match:
                val = match.group(1).strip()
                if val:
                    return f"न्यायाधीश: {val}"
        elif category == "petitioner":
            parties = header_item.get("parties", {})
            appellant = parties.get("appellant") if isinstance(parties, dict) else ""
            if appellant and appellant.strip():
                return f"पुनरावेदक/निवेदक: {appellant}"
            match = re.search(r"(?:पुनरावेदक|निवेदक)\s*[:：]\s*([^\n]+)", header_item.get("content", ""))
            if match:
                val = match.group(1).strip()
                if val:
                    return f"पुनरावेदक/निवेदक: {val}"
        elif category == "respondent":
            parties = header_item.get("parties", {})
            respondent = parties.get("respondent") if isinstance(parties, dict) else ""
            if respondent and respondent.strip():
                return f"प्रत्यर्थी/विपक्षी: {respondent}"
            match = re.search(r"(?:प्रत्यर्थी|विपक्षी)\s*[:：]\s*([^\n]+)", header_item.get("content", ""))
            if match:
                val = match.group(1).strip()
                if val:
                    return f"प्रत्यर्थी/विपक्षी: {val}"
        elif category == "subject":
            subject = header_item.get("subject", "") or header_item.get("case_type", "")
            if subject and subject.strip():
                return f"विषय: {subject}"
        elif category == "date":
            date = header_item.get("date", "")
            if date and date.strip():
                return f"फैसला मिति: {date}"
        elif category == "final_order":
            final_order = header_item.get("final_order")
            if final_order and final_order != "UNKNOWN" and final_order.strip():
                return f"अन्तिम आदेश: {final_order}"
            content = header_item.get("content", "")
            match = re.search(r"अन्तिम\s*निर्णय/आदेश\s*[:：]\s*([^\n]+)", content)
            if match:
                val = match.group(1).strip()
                if val:
                    return f"अन्तिम आदेश: {val}"
            if manual and manual.get("final_order"):
                val = manual['final_order']
                if val and val.strip():
                    return f"अन्तिम आदेश: {val}"
        elif category == "provisions":
            provisions = header_item.get("provisions", "")
            if provisions and provisions.strip():
                return f"प्रमुख कानूनी प्रावधान: {provisions}"
        elif category == "lawyers":
            al = header_item.get("appellant_lawyer", "")
            rl = header_item.get("respondent_lawyer", "")
            parts = []
            if al and al.strip():
                parts.append(f"पुनरावेदकका कानून व्यवसायी: {al}")
            if rl and rl.strip():
                parts.append(f"प्रत्यर्थीका कानून व्यवसायी: {rl}")
            if parts:
                return "\n".join(parts)

    # ========================================================================
    # CONTENT‑BASED EXTRACTION (NEW – scans all retrieved text)
    # ========================================================================
    # For petitioner/respondent
    if category == "petitioner":
        patterns = [
            r"निवेदक\s*[:：]\s*([^\n]+)",
            r"पुनरावेदक\s*[:：]\s*([^\n]+)",
            r"निवेदक\s*M\s*([^\n]+)",
            r"पुनरावेदक/निवेदक\s*[:：]\s*([^\n]+)",
            r"([^\n]+)\s*निवेदक",
        ]
        for pat in patterns:
            m = re.search(pat, all_content, re.I)
            if m:
                val = m.group(1).strip()
                if val and val not in ["प्रत्यर्थी/विपक्षी:", ""]:
                    return f"पुनरावेदक/निवेदक: {val}"
    elif category == "respondent":
        patterns = [
            r"प्रत्यर्थी\s*[:：]\s*([^\n]+)",
            r"विपक्षी\s*[:：]\s*([^\n]+)",
            r"प्रत्यर्थी/विपक्षी\s*[:：]\s*([^\n]+)",
        ]
        for pat in patterns:
            m = re.search(pat, all_content, re.I)
            if m:
                val = m.group(1).strip()
                if val and val not in ["पुनरावेदक/निवेदक:", ""]:
                    return f"प्रत्यर्थी/विपक्षी: {val}"

    # For final_order, scan for keywords in context
    if category == "final_order":
        for term in ["सदर", "उल्टी", "खारेज", "अमान्य", "बदर"]:
            if term in all_content:
                context = re.search(r"[^।]*{}[^।]*।".format(term), all_content)
                if context:
                    return f"अन्तिम आदेश: {context.group(0).strip()}"
                else:
                    return f"अन्तिम आदेश: {term}"
        m = re.search(r"(?:अन्तिम\s*आदेश|निर्णय)\s*[:：]\s*([^\n]+)", all_content)
        if m:
            return f"अन्तिम आदेश: {m.group(1).strip()}"

    # For legal_principle, scan for "सिद्धान्त" or "प्रतिपादन"
    if any(kw in q_lower for kw in ["सिद्धान्त", "प्रतिपादन", "precedent"]):
        m = re.search(r"(?:सिद्धान्त|प्रतिपादन)\s*[:：]\s*([^\n]+)", all_content)
        if m:
            return f"मुख्य कानूनी सिद्धान्त: {m.group(1).strip()}"

    # For provisions, if not already returned
    if category == "provisions":
        provisions = header_item.get("provisions", "") if header_item else ""
        if provisions and provisions.strip():
            return f"प्रमुख कानूनी प्रावधान: {provisions}"
        # Fallback: find धारा/दफा list
        matches = re.findall(r"(?:धारा|दफा)\s*[०-९0-9]+", all_content)
        if matches:
            unique = list(dict.fromkeys(matches))[:5]
            return "प्रमुख कानूनी प्रावधान: " + ", ".join(unique)

    # If nothing found, return None to invoke LLM
    return None

# ========================================================================
# GROQ CALL
# ========================================================================
def _call_groq(messages: list, model: str, max_tokens: int, stream: bool = False):
    client = get_groq_client()
    if not client:
        return None
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.15,
                max_tokens=max_tokens,
                stream=stream,
            )
        except RateLimitError:
            wait = 2 ** attempt
            print(f"[GROQ] Rate limited. Retrying in {wait}s...")
            time.sleep(wait)
        except APIStatusError as e:
            print(f"[GROQ] API error: {e}")
            break
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            break
    return None

# ========================================================================
# MAIN GENERATION
# ========================================================================
def _render_manual_summary(manual: dict) -> str:
    return f"""**मुद्दाको परिचय**
{manual['introduction']}

**पक्षकारहरू**
{manual['parties']}

**कानून व्यवसायीहरू**
{manual['lawyers']}

**मुख्य तथ्य**
{manual['key_facts']}

**मुख्य कानूनी प्रश्न**
{manual['legal_questions']}

**अदालतको तर्क/निष्कर्ष**
{manual['reasoning']}

**अन्तिम निर्णय/आदेश**
{manual['final_order']}

**मुख्य कानूनी सिद्धान्त**
{manual['legal_principle']}"""


def _render_case_about(manual: dict, fallback_subject: str = "") -> str:
    intro = (manual.get("introduction") or "").strip()
    facts = (manual.get("key_facts") or "").strip()
    subject = clean_ocr_field(manual.get("subject") or fallback_subject)
    first_fact = facts.split("।")[0].strip() if facts else ""
    if first_fact and not first_fact.endswith("।"):
        first_fact += "।"
    parts = []
    if subject:
        parts.append(f"यो **{subject}** सम्बन्धी मुद्दा हो।")
    elif intro:
        parts.append(intro)
    if first_fact:
        parts.append(first_fact)
    return " ".join(parts) if parts else intro


def generate_nepali_answer(
    query: str,
    retrieved_items: list,
    model_name: str = DEFAULT_MODEL,
    current_case: dict = None,
    metadata_info: dict = None,
    comparison_mode: bool = False,
    detected_numbers: list = None,
    stream: bool = False,
) -> str:
    deterministic = get_deterministic_answer(query)
    if deterministic and "निर्णय" not in (query or ""):
        return deterministic

    intent = detect_query_intent(query)
    case_id = current_case.get("case_id") if current_case else None
    dec_no = current_case.get("decision_no") if current_case else None

    if intent == "LIST_CASES":
        if metadata_info and "case_metadata" in metadata_info:
            cases = metadata_info["case_metadata"]
            if cases:
                lines = ["मसँग निम्न मुद्दाहरूको जानकारी छ:\n"]
                for cid, info in cases.items():
                    decision_no = info.get("decision_no", "अज्ञात")
                    date = info.get("date", "मिति उपलब्ध छैन")
                    subject = clean_ocr_field(info.get("subject", "विषय उपलब्ध छैन"))
                    lines.append(f"- निर्णय नं. {decision_no} (मिति: {date}, विषय: {subject})")
                return "\n".join(lines)
        return "मसँग हाल कुनै मुद्दाको जानकारी उपलब्ध छैन।"

    if dec_no and not decision_exists(dec_no, None) and case_id not in MANUAL_SUMMARIES:
        available = []
        if metadata_info and metadata_info.get("case_metadata"):
            available = [info.get("decision_no") for info in metadata_info["case_metadata"].values()]
        extra = f" उपलब्ध निर्णय नं.: {', '.join(available)}।" if available else ""
        return f"निर्णय नं. {dec_no} यस ज्ञानकोषमा अनुक्रमित छैन।{extra}"

    if not retrieved_items and case_id not in MANUAL_SUMMARIES:
        return NO_INFO

    manual = MANUAL_SUMMARIES.get(case_id) if case_id else None

    if intent == "CASE_ABOUT" and manual:
        subject = ""
        if metadata_info:
            subject = (metadata_info.get("case_metadata") or {}).get(case_id, {}).get("subject", "")
        return _render_case_about(manual, subject)

    if intent == "LEGAL_PRINCIPLE" and manual and manual.get("legal_principle"):
        return f"**मुख्य कानूनी सिद्धान्त**\n{manual['legal_principle']}"

    if intent == "CASE_SUMMARY" and manual:
        required = ["introduction", "parties", "lawyers", "key_facts", "legal_questions", "reasoning", "final_order", "legal_principle"]
        if all(k in manual for k in required):
            return _render_manual_summary(manual)

    factual_answer = answer_factual_query(query, retrieved_items or [], current_case)
    if factual_answer:
        return factual_answer

    # Filter to current case (with fallback)
    if not comparison_mode and current_case and case_id:
        filtered_items = [item for item in retrieved_items if item.get("case_id") == case_id]
        if filtered_items:
            retrieved_items = filtered_items
        # keep original if none

    # Prakaran extraction
    paragraph_match = re.search(r"(?:प्रकरण|अनुच्छेद|धारा)\s*नं\.?\s*([०-९0-9]+)", query)
    if paragraph_match:
        para_no = paragraph_match.group(1)
        for item in retrieved_items:
            if item.get("prakaran_no") == para_no:
                return f"प्रकरण/अनुच्छेद नं. {para_no} को पाठ:\n\n{item.get('content', '')}"
        return f"प्रकरण/अनुच्छेद नं. {para_no} को जानकारी उपलब्ध छैन।"

    # Build context for LLM
    chunks_to_use = retrieved_items[:5]
    evidence_parts = []
    for item in chunks_to_use:
        page = item.get("page", "?")
        content = item.get("content", "")
        if len(content) > 600:
            content = truncate_at_sentence(content, 600)
        evidence_parts.append(f"--- पृष्ठ {page} ---\n{content}")
    context = "\n\n".join(evidence_parts)

    manual_context = ""
    if case_id and case_id in MANUAL_SUMMARIES:
        manual = MANUAL_SUMMARIES[case_id]
        summary_fields = [
            f"**परिचय:** {manual.get('introduction', '')}",
            f"**पक्षकार:** {manual.get('parties', '')}",
            f"**न्यायाधीश:** {manual.get('judges', '')}",
            f"**मुख्य तथ्य:** {manual.get('key_facts', '')}",
            f"**अन्तिम आदेश:** {manual.get('final_order', '')}",
            f"**मुख्य कानूनी सिद्धान्त:** {manual.get('legal_principle', '')}",
            f"**अघिल्ला नजिर:** {manual.get('precedents', '')}",
        ]
        manual_context = "\n\n".join(f for f in summary_fields if f.strip())

    if not get_groq_client():
        return GROQ_UNAVAILABLE

    token_limit = 500 if intent in ["LIST_CASES", "CASE_LOOKUP"] else (1200 if intent == "CASE_SUMMARY" else 1024)

    system_prompt = f"""तपाईं नेपाली कानूनी सहायक हुनुहुन्छ। उत्तर नेपालीमा दिनुहोस्। 
**कडा नियम:** 
1. केवल तल दिइएको "प्रमाण" भित्रको जानकारी मात्र प्रयोग गर्नुहोस्।
2. यदि प्रमाणमा जानकारी छैन भने "उपलब्ध प्रमाणमा यस विषयको उल्लेख छैन।" भन्नुहोस्।
3. कुनै पनि तथ्य नबनाउनुहोस् (NO HALLUCINATION)।

हालको मुद्दा: {case_id if case_id else "अज्ञात"}

{manual_context if manual_context else ""}

प्रमाण (कागजातका अंशहरू):
{context}

प्रश्न: {query}

निर्देशन:
- यदि प्रश्नले अन्तिम आदेश सोधेको छ भने, प्रमाणहरूमा "सदर", "उल्टी", "खारेज", "अमान्य" जस्ता शब्दहरू खोजी निष्कर्ष निकाल्नुहोस्।
- यदि प्रश्नले कानूनी सिद्धान्त सोधेको छ भने, प्रमाणहरूमा "सिद्धान्त", "प्रतिपादन", "ठहर" जस्ता शब्दहरू खोजी उद्धृत गर्नुहोस्।
- यदि प्रश्नले अघिल्ला नजिर सोधेको छ भने, प्रमाणहरूमा "नजीर", "पूर्व निर्णय", "case citation" खोजी उल्लेख गर्नुहोस्।
- यदि जानकारी छैन भने स्पष्ट रूपमा भन्नुहोस् कि "उपलब्ध प्रमाणमा यस विषयको उल्लेख छैन।"
- उत्तरलाई संक्षिप्त र स्पष्ट राख्नुहोस्।"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": query.strip()},
    ]

    response = _call_groq(messages, model_name, token_limit, stream=stream)
    if response is None:
        return GROQ_UNAVAILABLE

    if stream:
        return response

    raw_answer = response.choices[0].message.content
    if raw_answer and raw_answer.strip():
        return clean_and_repair_nepali_output(raw_answer)

    # Plain text fallback from header
    for item in retrieved_items:
        if item.get("is_header", False):
            content = item.get("content", "")
            if "लायर" in query or "कानून व्यवसायी" in query:
                match = re.search(r"पुनरावेदकका कानून व्यवसायी:\s*([^\n]+)", content)
                if match:
                    val = match.group(1).strip()
                    if val:
                        return f"पुनरावेदकका कानून व्यवसायी: {val}"
            if "न्यायाधीश" in query or "इजलास" in query:
                match = re.search(r"न्यायाधीश:\s*([^\n]+)", content)
                if match:
                    val = match.group(1).strip()
                    if val:
                        return f"न्यायाधीश: {val}"
            return "उपलब्ध कागजातमा यस प्रश्नको जानकारी छैन। कृपया अर्को प्रश्न सोध्नुहोस्।"

    return NO_INFO