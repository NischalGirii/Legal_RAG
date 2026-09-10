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
from src.hybrid_search import detect_query_intent, chunks_for_decision, decision_exists, snap_decision_number

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

def get_deterministic_answer(query: str) -> str | None:
    q = query.lower()
    q = re.sub(r"[^\w\s\u0900-\u097f]", "", q)

    if "उत्प्रेषण" in q and ("के हो" in q or "बारे" in q):
        return "उत्प्रेषण (Certiorari) भनेको तल्लो अदालत वा निकायको आदेश, निर्णय वा कार्यको पुनरावलोकन गर्न माग गरिएको रिट हो। यसले अधिकारक्षेत्र बाहिर गएको वा कानूनी त्रुटि भएको कार्यलाई बदर गर्न माग गरिन्छ।"
    if "परमादेश" in q and ("के हो" in q or "बारे" in q):
        return "परमादेश (Mandamus) भनेको कुनै सार्वजनिक निकायलाई आफ्नो कानूनी वा संवैधानिक कर्तव्य पालना गर्न आदेश दिन माग गरिएको रिट हो।"
    if "बन्दीप्रत्यक्षीकरण" in q and ("के हो" in q or "बारे" in q):
        return "बन्दीप्रत्यक्षीकरण (Habeas Corpus) भनेको कुनै व्यक्तिलाई गैरकानूनी रूपमा हिरासतमा राखिएको भए उसलाई अदालत समक्ष उपस्थित गराउन माग गरिएको रिट हो।"
    if "न्यायिक पुनरावलोकन" in q and ("के हो" in q or "बारे" in q):
        return "न्यायिक पुनरावलोकन (Judicial Review) भनेको अदालतले विधायिका वा कार्यपालिकाका कार्यहरू संविधानसँग बाझिएको छ कि छैन भनी परीक्षण गर्ने अधिकार हो।"
    if "रिट" in q and ("के हो" in q or "बारे" in q):
        return "रिट (Writ) भनेको अदालतबाट जारी गरिने लिखित आदेश हो, जसले कानूनी कर्तव्य पालना गर्न, मौलिक हक संरक्षण गर्न, वा गैरकानूनी कार्य रोक्न निर्देशन दिन्छ।"
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
    dec_no = snap_decision_number(str(dec_no), chunk_metadata)
    case_id = f"decision_{dec_no}"
    meta_cases = (metadata_info or {}).get("case_metadata") or {}
    ingest = meta_cases.get(case_id) or meta_cases.get(dec_no) or {}
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
    return "\n\n".join(blocks)

def generate_comparison_answer(query: str, decision_numbers: list, metadata_info: dict,
                               chunk_metadata: list, collection, model, bm25,
                               top_k=5, alpha=0.15) -> str | None:
    if not decision_numbers or len(decision_numbers) < 2:
        return None

    unique_numbers = list(dict.fromkeys(snap_decision_number(str(n), chunk_metadata) for n in decision_numbers))
    cards = [build_case_card(n, metadata_info, chunk_metadata) for n in unique_numbers]
    return _format_structured_comparison(cards)

def _extract_from_manual(manual: dict, category: str) -> str | None:
    parties = manual.get("parties", "")
    intro = manual.get("introduction", "")

    if category == "judges":
        judges = clean_ocr_field(manual.get("judges", ""))
        if judges:
            return f"न्यायाधीश: {judges}"
        bench = re.search(r"((?:सम्माननीय|माननीय|प्रधानन्यायाधीश|न्यायाधीश).{8,220}इजलास)", intro)
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
    if category == "subject":
        subject = manual.get("subject", "") or manual.get("case_type", "")
        if subject and subject.strip():
            return f"विषय: {subject.strip()}"
        return None
    return None

def answer_factual_query(query: str, retrieved_items: list, current_case: dict = None) -> str | None:
    if not retrieved_items:
        return None

    q_lower = query.lower()
    header_item = next((item for item in retrieved_items if item.get("is_header")), None)
    all_content = "\n".join([item.get("content", "") for item in retrieved_items])

    if header_item:
        if any(kw in q_lower for kw in ["अन्तिम आदेश", "निष्कर्ष", "सदर", "उल्टी", "खारेज"]):
            val = clean_ocr_field(header_item.get("final_order"))
            if val and val != "UNKNOWN":
                return f"अन्तिम आदेश: {val}"
        if any(kw in q_lower for kw in ["सिद्धान्त", "प्रतिपादन", "precedent"]):
            val = clean_ocr_field(header_item.get("legal_principle"))
            if val and val != "UNKNOWN":
                return f"मुख्य कानूनी सिद्धान्त: {val}"

    category = None
    if any(kw in q_lower for kw in ["कानून व्यवसायी", "अधिवक्ता", "वकील", "बहस गर्ने"]):
        category = "lawyers"
    elif ("निवेदक" in q_lower or "पुनरावेदक" in q_lower) and "विपक्षी" not in q_lower:
        category = "petitioner"
    elif "विपक्षी" in q_lower or "प्रत्यर्थी" in q_lower:
        category = "respondent"
    elif any(kw in q_lower for kw in ["न्यायाधीश", "इजलास", "बेन्च"]):
        category = "judges"
    elif "विषय" in q_lower or "प्रकार" in q_lower:
        category = "subject"
    elif "मिति" in q_lower or "कहिले" in q_lower:
        category = "date"
    elif ("अन्तिम" in q_lower and "आदेश" in q_lower) or "निष्कर्ष" in q_lower:
        category = "final_order"

    if category is None:
        return None

    case_id = current_case.get("case_id") if current_case else None
    manual = MANUAL_SUMMARIES.get(case_id) if case_id else None
    if manual:
        ans = _extract_from_manual(manual, category)
        if ans and ans.strip():
            return ans

    if header_item:
        if category == "judges":
            judges = clean_ocr_field(header_item.get("judges", ""))
            if judges:
                return f"न्यायाधीश: {judges}"
        elif category == "petitioner":
            appellant = (header_item.get("parties") or {}).get("appellant")
            if appellant:
                return f"पुनरावेदक/निवेदक: {appellant}"
        elif category == "respondent":
            respondent = (header_item.get("parties") or {}).get("respondent")
            if respondent:
                return f"प्रत्यर्थी/विपक्षी: {respondent}"
        elif category == "subject":
            subject = header_item.get("subject") or header_item.get("case_type")
            if subject:
                return f"विषय: {subject}"
        elif category == "date":
            date = header_item.get("date")
            if date:
                return f"फैसला मिति: {date}"
        elif category == "final_order":
            final_order = header_item.get("final_order")
            if final_order and final_order != "UNKNOWN":
                return f"अन्तिम आदेश: {final_order}"

    return None

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

def _render_manual_summary(manual: dict) -> str:
    return f"""**मुद्दाको परिचय**
{manual.get('introduction', '')}

**पक्षकारहरू**
{manual.get('parties', '')}

**मुख्य तथ्य**
{manual.get('key_facts', '')}

**अन्तिम निर्णय/आदेश**
{manual.get('final_order', '')}

**मुख्य कानूनी सिद्धान्त**
{manual.get('legal_principle', '')}"""

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
            time.sleep(2 ** attempt)
        except Exception:
            break
    return None

def generate_nepali_answer(
    query: str,
    retrieved_items: list,
    model_name: str = DEFAULT_MODEL,
    current_case: dict = None,
    metadata_info: dict = None,
    comparison_mode: bool = False,
    detected_numbers: list = None,
    stream: bool = False,
    chunk_metadata: list = None,
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
                lines = ["मसँग निम्न मुद्दा/निर्णयहरूको जानकारी उपलब्ध छ:\n"]
                seen_decisions = set()
                for cid, info in cases.items():
                    d_no = str(info.get("decision_no") or "").strip()
                    if not d_no or d_no in seen_decisions:
                        continue
                    seen_decisions.add(d_no)
                    date = info.get("date", "उपलब्ध छैन")
                    subject = clean_ocr_field(info.get("subject", "उपलब्ध छैन"))
                    lines.append(f"- निर्णय नं. {d_no} (विषय: {subject}, मिति: {date})")
                return "\n".join(lines)
        return "मसँग हाल कुनै मुद्दाको जानकारी उपलब्ध छैन।"

    # Use the real chunk_metadata (when the caller passes it) instead of
    # forcing this check to rely solely on the case_index.json file. This
    # keeps this check consistent with every other decision_exists() call
    # in the pipeline (perform_hybrid_search / main.py already pass real
    # chunk_metadata), so a decision present in the vector index but not
    # yet reflected in case_index.json isn't wrongly reported as unindexed.
    if dec_no and not decision_exists(dec_no, chunk_metadata) and case_id not in MANUAL_SUMMARIES:
        available = []
        if metadata_info and metadata_info.get("case_metadata"):
            available = list(dict.fromkeys(
                str(info.get("decision_no")) for info in metadata_info["case_metadata"].values() if info.get("decision_no")
            ))
        extra = f" उपलब्ध निर्णय नं.: {', '.join(available)}।" if available else ""
        return f"निर्णय नं. {dec_no} यस ज्ञानकोषमा अनुक्रमित छैन।{extra}"

    if not retrieved_items and case_id not in MANUAL_SUMMARIES:
        return NO_INFO

    manual = MANUAL_SUMMARIES.get(case_id) if case_id else None

    # Handle summary / case about
    if intent in ("CASE_ABOUT", "CASE_LOOKUP") and manual:
        subject = ""
        if metadata_info:
            subject = (metadata_info.get("case_metadata") or {}).get(case_id, {}).get("subject", "")
        return _render_case_about(manual, subject)

    if intent == "CASE_SUMMARY" and manual:
        return _render_manual_summary(manual)

    if intent == "LEGAL_PRINCIPLE" and manual and manual.get("legal_principle"):
        return f"**मुख्य कानूनी सिद्धान्त**\n{manual['legal_principle']}"

    factual_answer = answer_factual_query(query, retrieved_items or [], current_case)
    if factual_answer:
        return factual_answer

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
        man = MANUAL_SUMMARIES[case_id]
        summary_fields = [
            f"**परिचय:** {man.get('introduction', '')}",
            f"**पक्षकार:** {man.get('parties', '')}",
            f"**न्यायाधीश:** {man.get('judges', '')}",
            f"**मुख्य तथ्य:** {man.get('key_facts', '')}",
            f"**अन्तिम आदेश:** {man.get('final_order', '')}",
            f"**मुख्य कानूनी सिद्धान्त:** {man.get('legal_principle', '')}",
        ]
        manual_context = "\n\n".join(f for f in summary_fields if f.strip())

    if not get_groq_client():
        return GROQ_UNAVAILABLE

    system_prompt = f"""तपाईं नेपाली कानूनी सहायक हुनुहुन्छ। उत्तर शुद्ध नेपालीमा दिनुहोस्।

**कडा नियम:**
1. तल दिइएको "प्रमाण" भित्रको जानकारी मात्र प्रयोग गर्नुहोस्।
2. प्रमाणमा उल्लेख भएको तथ्य, कानूनी आधार वा अदालतको ठहर मात्र उल्लेख गर्नुहोस्।
3. यदि प्रमाणमा जानकारी छैन भने "उपलब्ध प्रमाणमा यस विषयको उल्लेख छैन।" भन्नुहोस्।
4. आफ्ना तर्फबाट कुनै पनि काल्पनिक तथ्य नथप्नुहोस्।

हालको मुद्दा: {case_id if case_id else "अज्ञात"}

{manual_context}

प्रमाण (कागजातका अंशहरू):
{context}

प्रश्न: {query}
उत्तर:"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": query.strip()},
    ]

    response = _call_groq(messages, model_name, 1024, stream=stream)
    if response is None:
        return GROQ_UNAVAILABLE

    if stream:
        return response

    raw_answer = response.choices[0].message.content
    if raw_answer and raw_answer.strip():
        return clean_and_repair_nepali_output(raw_answer)

    return NO_INFO
