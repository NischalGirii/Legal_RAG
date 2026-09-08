import os
import re
import time
import json
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv
from src.text_processor import clean_and_repair_nepali_output, normalize_digits, clean_ocr_field
from src.config import CASE_INDEX_PATH, INGEST_METADATA_PATH, LLM_MODEL
from src.hybrid_search import detect_query_intent, chunks_for_decision, decision_exists, snap_decision_number, get_case_index

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_groq_client = None
DEFAULT_MODEL = LLM_MODEL

def get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

NO_INFO = "माफ गर्नुहोस्, उपलब्ध प्रमाण वा कागजातहरूमा यस विषयको उल्लेख छैन।"
GROQ_UNAVAILABLE = "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"
SERVER_ERROR = "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"

def get_deterministic_answer(query: str) -> str | None:
    q = query.lower()
    q = re.sub(r"[^\w\s\u0900-\u097f]", "", q)
    if "उत्प्रेषण" in q and ("के हो" in q or "बारे" in q or "परिभाषा" in q):
        return "उत्प्रेषण (Certiorari) भनेको अधिकारक्षेत्र नाघेर वा प्राकृतिक न्यायको सिद्धान्त विपरीत गरिएको तल्लो अदालत वा प्रशासनिक निकायको गैरकानूनी निर्णय, आदेश वा कामकारबाहीलाई बदर गर्न माग गरिने असाधारण अधिकारक्षेत्रअन्तर्गतको रिट हो।"
    if "परमादेश" in q and ("के हो" in q or "बारे" in q or "परिभाषा" in q):
        return "परमादेश (Mandamus) भनेको कुनै सार्वजनिक पदाधिकारी वा निकायलाई आफ्नो कानूनी वा संवैधानिक कर्तव्य पालना गर्न अदालतबाट जारी गरिने निर्देशनात्मक आदेश हो।"
    if "बन्दीप्रत्यक्षीकरण" in q and ("के हो" in q or "बारे" in q or "परिभाषा" in q):
        return "बन्दीप्रत्यक्षीकरण (Habeas Corpus) भनेको कुनै व्यक्तिलाई गैरकानूनी रूपमा थुनामा वा नियन्त्रणमा राखिएको अवस्थामा निजलाई अदालतसमक्ष उपस्थित गराई थुनामुक्त गर्न जारी गरिने रिट हो।"
    return None

def build_case_card(dec_no: str, metadata_info: dict, chunk_metadata: list) -> dict:
    dec_no = snap_decision_number(str(dec_no), chunk_metadata)
    case_id = f"decision_{dec_no}"
    meta_cases = (metadata_info or {}).get("case_metadata") or {}
    ingest = meta_cases.get(dec_no) or meta_cases.get(case_id) or {}
    chunks = chunks_for_decision(chunk_metadata, dec_no, case_id)
    header = next((c for c in chunks if c.get("is_header")), chunks[0] if chunks else {})
    petitioner = (
        ingest.get("parties", {}).get("appellant") if isinstance(ingest.get("parties"), dict)
        else header.get("parties", {}).get("appellant") if isinstance(header.get("parties"), dict)
        else header.get("parties") or ""
    )
    respondent = (
        ingest.get("parties", {}).get("respondent") if isinstance(ingest.get("parties"), dict)
        else header.get("parties", {}).get("respondent") if isinstance(header.get("parties"), dict)
        else ""
    )
    return {
        "decision_no": dec_no,
        "case_id": case_id,
        "known": bool(chunks or ingest),
        "source": ingest.get("source") or header.get("source") or "",
        "date": clean_ocr_field(ingest.get("date") or header.get("date") or ""),
        "court": clean_ocr_field(ingest.get("court") or header.get("court") or "सर्वोच्च अदालत"),
        "subject": clean_ocr_field(ingest.get("subject") or header.get("subject") or ""),
        "judges": clean_ocr_field(ingest.get("judges") or header.get("judges") or ""),
        "petitioner": clean_ocr_field(petitioner),
        "respondent": clean_ocr_field(respondent),
        "appellant_lawyer": clean_ocr_field(ingest.get("appellant_lawyer") or header.get("appellant_lawyer") or ""),
        "respondent_lawyer": clean_ocr_field(ingest.get("respondent_lawyer") or header.get("respondent_lawyer") or ""),
        "provisions": clean_ocr_field(ingest.get("provisions") or header.get("provisions") or ""),
        "final_order": clean_ocr_field(ingest.get("final_order") or header.get("final_order") or ""),
        "legal_principle": clean_ocr_field(ingest.get("legal_principle") or header.get("legal_principle") or ""),
        "precedents": clean_ocr_field(ingest.get("precedents") or header.get("precedents") or ""),
    }

def _format_structured_comparison(cards: list[dict]) -> str:
    blocks = []
    for card in cards:
        if not card.get("known"):
            blocks.append(f"### निर्णय नं. {card['decision_no']}\nयस ज्ञानकोषमा यो निर्णय अनुक्रमित छैन।")
            continue
        blocks.append(
            f"### निर्णय नं. {card['decision_no']}\n"
            f"* **मुद्दा/विषय:** {card['subject'] or 'उपलब्ध छैन'}\n"
            f"* **फैसला मिति:** {card['date'] or 'उपलब्ध छैन'}\n"
            f"* **इजलास/न्यायाधीश:** {card['judges'] or 'उपलब्ध छैन'}\n"
            f"* **पुनरावेदक/निवेदक:** {card['petitioner'] or 'उपलब्ध छैन'}\n"
            f"* **प्रत्यर्थी/विपक्षी:** {card['respondent'] or 'उपलब्ध छैन'}\n"
            f"* **प्रमुख कानूनी प्रावधान:** {card['provisions'] or 'उपलब्ध छैन'}\n"
            f"* **अदालतको ठहर / अन्तिम आदेश:** {card['final_order'] or 'उपलब्ध छैन'}\n"
            f"* **मुख्य कानूनी सिद्धान्त:** {card['legal_principle'] or 'उपलब्ध छैन'}"
        )
    return "\n\n".join(blocks)

def generate_comparison_answer(
    query: str,
    decision_numbers: list,
    metadata_info: dict,
    chunk_metadata: list,
    collection,
    model,
    bm25,
    top_k: int = 5,
    alpha: float = 0.5,
) -> str | None:
    if not decision_numbers or len(decision_numbers) < 2:
        return None

    unique_numbers = list(dict.fromkeys(snap_decision_number(str(n), chunk_metadata) for n in decision_numbers))
    cards = [build_case_card(n, metadata_info, chunk_metadata) for n in unique_numbers]
    return _format_structured_comparison(cards)

def _call_groq(messages: list, model: str, max_tokens: int = 1500, stream: bool = False):
    client = get_groq_client()
    if not client:
        return None
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.1,
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
                lines = ["मसँग निम्न मुद्दा/निर्णयहरूको आधिकारिक जानकारी उपलब्ध छ:\n"]
                seen_decisions = set()
                for cid, info in sorted(cases.items()):
                    d_no = str(info.get("decision_no") or "").strip()
                    if not d_no or d_no in seen_decisions:
                        continue
                    seen_decisions.add(d_no)
                    date = info.get("date", "उपलब्ध छैन")
                    subject = clean_ocr_field(info.get("subject", "उपलब्ध छैन"))
                    final_order = clean_ocr_field(info.get("final_order", ""))
                    order_text = f", अन्तिम आदेश: {final_order}" if final_order else ""
                    lines.append(f"- **निर्णय नं. {d_no}**: {subject} (फैसला मिति: {date}{order_text})")
                return "\n".join(lines)
        return "मसँग हाल कुनै मुद्दाको जानकारी उपलब्ध छैन।"

    if dec_no and not decision_exists(dec_no, chunk_metadata):
        available = []
        if metadata_info and metadata_info.get("case_metadata"):
            available = sorted(set(
                str(info.get("decision_no")) for info in metadata_info["case_metadata"].values() if info.get("decision_no")
            ))
        extra = f" उपलब्ध निर्णय नं.: {', '.join(available)}।" if available else ""
        return f"निर्णय नं. {dec_no} यस ज्ञानकोषमा अनुक्रमित छैन।{extra}"

    if not retrieved_items:
        return NO_INFO

    # Build evidence with metadata-based citations
    evidence_parts = []
    for idx, item in enumerate(retrieved_items, start=1):
        source = item.get("source", "PDF")
        page = item.get("page", "?")
        d_no = item.get("decision_no", "")
        d_no_orig = item.get("decision_no_original", d_no)
        prakaran = item.get("prakaran_no")
        prakaran_info = f", प्रकरण नं. {prakaran}" if prakaran else ""
        content = item.get("content", "").strip()
        evidence_parts.append(
            f"--- [प्रमाण खण्ड #{idx} | स्रोत: {source}, निर्णय नं.: {d_no_orig}, पृष्ठ: {page}{prakaran_info}] ---\n{content}"
        )
    context = "\n\n".join(evidence_parts)

    case_context = ""
    if dec_no:
        card = build_case_card(dec_no, metadata_info, chunk_metadata or [])
        if card.get("known"):
            case_context = (
                f"मुद्दाको आधिकारिक विवरण:\n"
                f"- निर्णय नं.: {card['decision_no']}\n"
                f"- विषय: {card['subject']}\n"
                f"- मिति: {card['date']}\n"
                f"- इजलास / न्यायाधीश: {card['judges']}\n"
                f"- पक्षकार (निवेदक/पुनरावेदक): {card['petitioner']}\n"
                f"- पक्षकार (विपक्षी/प्रत्यर्थी): {card['respondent']}\n"
                f"- अन्तिम आदेश/ठहर: {card['final_order']}\n"
                f"- मुख्य सिद्धान्त: {card['legal_principle']}\n"
            )

    if not get_groq_client():
        return GROQ_UNAVAILABLE

    system_prompt = f"""तपाईं सर्वोच्च अदालत नेपालका फैसलाहरूको आधिकारिक नेपाली कानूनी विश्लेषक तथा RAG सहायक हुनुहुन्छ।

**कडा नियम र निर्देशनहरू:**
1. तल प्रस्तुत गरिएको "प्रमाण खण्ड" (Retrieved Context) र "मुद्दाको आधिकारिक विवरण" मा मात्र पूर्ण रूपमा आधारित भएर शुद्ध नेपाली भाषामा तथ्यपरक उत्तर दिनुहोस्।
2. प्रमाणमा स्पष्ट उल्लेख भएका मिति, दफा, धारा, नियम, न्यायाधीशका नाम, पक्षकार, जरिवाना रकम तथा अदालतको अन्तिम ठहर ठ्याक्कै दुरुस्त लेख्नुहोस्।
3. **महत्त्वपूर्ण: कानूनी अधिकार र सो अधिकारको प्रयोग बीचको भिन्नता बुझ्नुहोस्।**
   - यदि कुनै कानूनले नागरिकता रद्द गर्न "सकिने" व्यवस्था गरेको छ भने त्यो **अधिकार** हो।
   - यदि त्यो अधिकार **यस मुद्दामा सही तरिकाले प्रयोग भयो कि भएन** भन्ने प्रश्नको उत्तर दिनुपर्ने हो भने, प्रमाणमा अदालतको **ठहर** (तसर्थ, अतः, बदर, उल्टी, सदर) हेर्नुहोस्।
   - ठहर नै अन्तिम निर्णय हो। ठहरलाई बेवास्ता गरी कानूनको सामान्य व्यवस्था मात्र उल्लेख गर्नु गलत हो।
4. आफ्ना तर्फबाट कुनै पनि काल्पनिक तथ्य वा अनुमान नथप्नुहोस् (Zero Hallucination)।
5. यदि सोधिएको प्रश्नको उत्तर दिइएको प्रमाणमा छैन भने स्पष्ट रूपमा भन्नुहोस्: "माफ गर्नुहोस्, उपलब्ध प्रमाण वा कागजातहरूमा यस विषयको उल्लेख छैन।"
6. **स्रोत उल्लेख गर्दा प्रमाण खण्डमा दिइएको निर्णय नं. र पृष्ठ मात्र प्रयोग गर्नुहोस्, आफ्नो तर्फबाट स्रोत नबनाउनुहोस्।**
7. जब प्रमाणले कुनै भूमिका वा निष्कर्षलाई स्पष्ट रूपमा “निर्णायक” भनेर उल्लेख नगरेको हो भने त्यस्तो शब्द प्रयोग नगर्नुहोस्। प्रमाणमा भएकै शब्द प्रयोग गर्नुहोस् (जस्तै: “प्रतिवेदन प्रस्तुत गरिएको थियो” भन्नुहोस्, “निर्णायक भूमिका निर्वाह” नभन्नुहोस्)।

{case_context}

प्रमाण खण्डहरू (Retrieved Evidence):
{context}
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": query.strip()},
    ]

    response = _call_groq(messages, model_name, max_tokens=1500, stream=stream)
    if response is None:
        return GROQ_UNAVAILABLE

    if stream:
        return response

    raw_answer = response.choices[0].message.content
    if raw_answer and raw_answer.strip():
        return clean_and_repair_nepali_output(raw_answer)

    return NO_INFO