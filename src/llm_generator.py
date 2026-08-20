import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
from src.text_processor import clean_and_repair_nepali_output
from src.hybrid_search import detect_query_intent

load_dotenv()

# ========================================================================
# REUSE GROQ CLIENT (instantiated once)
# ========================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ========================================================================
# LOAD MANUAL CASE SUMMARIES
# ========================================================================
MANUAL_SUMMARIES = {}
summary_path = "./models/case_summaries.json"
if os.path.exists(summary_path):
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            MANUAL_SUMMARIES = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load manual summaries: {e}")

# ========================================================================
# FALLBACK RESPONSES
# ========================================================================
NO_INFO = "माफ गर्नुहोस्, यस विषयमा उपलब्ध जानकारी छैन।"
GROQ_UNAVAILABLE = "माफ गर्नुहोस्, अहिले सूचना सेवा उपलब्ध छैन।"
SERVER_ERROR = "माफ गर्नुहोस्, अहिले सर्भरमा समस्या देखिएको छ।"

# ========================================================================
# HELPER: Truncate text at sentence boundary (for snippets)
# ========================================================================
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
# REFACTORED FACTUAL EXTRACTION (based on Gemini's analysis)
# ========================================================================
def answer_factual_query(query: str, retrieved_items: list, current_case: dict = None) -> str | None:
    """
    Fixed extraction logic with strict pattern hierarchy and word boundary guards.
    """
    if not retrieved_items:
        return None

    q_lower = query.lower()

    # ---- 1. EXCLUSIONS FOR LLM QA / REASONING QUERIES ----
    # If the user asks about reasoning, claims, or application of provisions, pass to LLM directly.
    llm_reasoning_triggers = ["तर्क", "निष्कर्ष", "दाबी", "खारेज", "सिद्धान्त", "कसरी", "के-के"]
    if any(trigger in q_lower for trigger in llm_reasoning_triggers) and not any(kw in q_lower for kw in ["को-को", "को हुन्", "को हुनुहुन्छ"]):
        # Check if it's asking for reasoning or application
        if "धारा" in q_lower or "दफा" in q_lower or "दाबी" in q_lower or "खारेज" in q_lower:
            return None

    # ---- 2. ORDERED CATEGORY PATTERNS (Specific -> General) ----
    # Lawyers and Judges MUST be checked BEFORE petitioner/respondent to avoid keyword hijacking.
    
    # LAWYERS
    if any(kw in q_lower for kw in ["कानून व्यवसायी", "अधिवक्ता", "वकील", "बहस गर्ने"]):
        matched_category = "lawyers"
    # JUDGES
    elif any(kw in q_lower for kw in ["न्यायाधीश", "इजलास", "बेन्च", "हेर्नुभएको"]):
        matched_category = "judges"
    # PROVISIONS
    elif any(kw in q_lower for kw in ["धारा", "दफा", "प्रावधान"]):
        matched_category = "provisions"
    # RESPONDENT
    elif any(kw in q_lower for kw in ["प्रत्यर्थी", "विपक्षी", "respondent"]):
        matched_category = "respondent"
    # PETITIONER (Strictly check context to avoid false positives)
    elif any(kw in q_lower for kw in ["निवेदक", "पुनरावेदक", "petitioner"]) or re.search(r"\bकसले\b", q_lower):
        matched_category = "petitioner"
    else:
        return None

    # ---- SPECIAL CASE: Provisions query handling ----
    if matched_category == "provisions":
        # Check for specific numbers (e.g. धारा ८८) or singular interrogatives (कुन धारा)
        specific_pattern = r"(?:धारा|दफा)\s*[०-९0-9]+"
        # Exclude plural "कुन-कुन" from triggering singular "कुन धारा"
        singular_interrogative = r"(?<!कुन-)कुन\s*(?:धारा|दफा)" 
        about_pattern = r"(?:धारा|दफा).*बारे"

        if (re.search(specific_pattern, query) or 
            re.search(singular_interrogative, query) or 
            re.search(about_pattern, query)):
            # Hand off specific/complex provision questions to LLM
            return None

    # ---- DATA EXTRACTION ----
    case_id = current_case.get("case_id") if current_case else None
    manual = MANUAL_SUMMARIES.get(case_id) if case_id else None

    # A. Try Manual Summary Extraction
    if manual:
        if matched_category == "petitioner":
            parties_str = manual.get("parties", "")
            if "निवेदक:" in parties_str:
                start = parties_str.find("निवेदक:") + len("निवेदक:")
                end = parties_str.find("विपक्षी:", start)
                if end == -1:
                    end = len(parties_str)
                petitioner = parties_str[start:end].strip().strip("।")
                if petitioner:
                    return f"यस रिट निवेदनमा निवेदक (पुनरावेदक) : {petitioner} हुनुहुन्छ।"

        elif matched_category == "respondent":
            parties_str = manual.get("parties", "")
            if "विपक्षी:" in parties_str:
                start = parties_str.find("विपक्षी:") + len("विपक्षी:")
                respondent = parties_str[start:].strip().strip("।")
                if respondent:
                    return f"यस रिट निवेदनमा विपक्षी (प्रत्यर्थी) : {respondent} हुनुहुन्छ।"

        elif matched_category == "judges":
            intro = manual.get("introduction", "")
            judge_match = re.search(r"(?:न्यायाधीशहरू|न्यायाधीश)\s*[:：]?\s*([^\n।]+)", intro)
            if judge_match:
                return f"यस मुद्दाको इजलासमा न्यायाधीशहरू: {judge_match.group(1).strip()} रहनुभएको थियो।"
            # Search in raw intro string
            lines = [l.strip() for l in intro.split("\n") if "न्यायाधीश" in l or "प्रधानन्यायाधीश" in l]
            if lines:
                return f"यस मुद्दाको इजलासमा न्यायाधीशहरू: {' '.join(lines)} रहनुभएको थियो।"

        elif matched_category == "lawyers":
            lawyers = manual.get("lawyers", "")
            if lawyers and lawyers.strip():
                return f"यस मुद्दामा कानून व्यवसायीहरू:\n{lawyers}"

        elif matched_category == "provisions":
            provisions = manual.get("provisions") or manual.get("key_provisions")
            if provisions:
                return f"यस मुद्दामा उल्लेखित प्रमुख कानूनी प्रावधानहरू: {provisions}"

    # B. Fallback to Header Item Metadata
    header_item = next((item for item in retrieved_items if item.get("is_header")), retrieved_items[0])
    
    if header_item:
        if matched_category == "petitioner":
            p = header_item.get("parties", {})
            pet = p.get("appellant") or p.get("petitioner") if isinstance(p, dict) else None
            if pet:
                return f"यस रिट निवेदनमा निवेदक (पुनरावेदक) : {pet} हुनुहुन्छ।"

        elif matched_category == "respondent":
            p = header_item.get("parties", {})
            resp = p.get("respondent") or p.get("defendant") if isinstance(p, dict) else None
            if resp:
                return f"यस रिट निवेदनमा विपक्षी (प्रत्यर्थी) : {resp} हुनुहुन्छ।"

        elif matched_category == "judges":
            judges = header_item.get("judges")
            if judges:
                return f"यस मुद्दाको इजलासमा न्यायाधीशहरू: {judges} रहनुभएको थियो।"

        elif matched_category == "lawyers":
            a_law = header_item.get("appellant_lawyer")
            r_law = header_item.get("respondent_lawyer")
            if a_law or r_law:
                return f"पुनरावेदकका कानून व्यवसायी: {a_law or 'उल्लेख नभएको'}\nप्रत्यर्थीका कानून व्यवसायी: {r_law or 'उल्लेख नभएको'}"

        elif matched_category == "provisions":
            provs = header_item.get("provisions")
            if provs:
                return f"यस मुद्दामा उल्लेखित प्रमुख कानूनी प्रावधानहरू: {provs}"

    return None

# ========================================================================
# MAIN GENERATION FUNCTION
# ========================================================================
def generate_nepali_answer(
    query: str,
    retrieved_items: list,
    model_name: str = "openai/gpt-oss-20b",
    current_case: dict = None,
    metadata_info: dict = None
) -> str:
    if not retrieved_items:
        return NO_INFO

    intent = detect_query_intent(query)
    case_id = current_case.get("case_id") if current_case else None

    # ---- LIST CASES ----
    if intent == "LIST_CASES":
        if metadata_info and "case_metadata" in metadata_info:
            cases = metadata_info["case_metadata"]
            if cases:
                lines = ["मसँग निम्न मुद्दाहरूको जानकारी छ:\n"]
                for case_id, info in cases.items():
                    decision_no = info.get("decision_no", "अज्ञात")
                    date = info.get("date", "मिति उपलब्ध छैन")
                    subject = info.get("subject", "विषय उपलब्ध छैन")
                    lines.append(f"- निर्णय नं. {decision_no} (मिति: {date}, विषय: {subject})")
                return "\n".join(lines)
            else:
                return "मसँग हाल कुनै मुद्दाको जानकारी उपलब्ध छैन।"
        else:
            return "मसँग हाल कुनै मुद्दाको जानकारी उपलब्ध छैन।"

    # ---- CASE SUMMARY: return manual summary if available ----
    if intent == "CASE_SUMMARY":
        if case_id and case_id in MANUAL_SUMMARIES:
            manual = MANUAL_SUMMARIES[case_id]
            if all(key in manual for key in ["introduction", "parties", "lawyers", "key_facts", "legal_questions", "reasoning", "final_order", "legal_principle"]):
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

    # ---- FACTUAL EXTRACTION (parties, judges, lawyers, provisions) ----
    factual_answer = answer_factual_query(query, retrieved_items, current_case)
    if factual_answer:
        return factual_answer

    # ---- FOR OTHER INTENTS: use only top 5 chunks to save tokens ----
    chunks_to_use = retrieved_items[:5]
    evidence_parts = []
    for item in chunks_to_use:
        page = item.get("page", "?")
        content = item.get("content", "")
        if len(content) > 600:
            content = truncate_at_sentence(content, 600)
        evidence_parts.append(f"--- पृष्ठ {page} ---\n{content}")
    context = "\n\n".join(evidence_parts)

    # ---- TRY GROQ ----
    if groq_client:
        system_prompt = f"""तपाईं नेपाली कानूनी सहायक हुनुहुन्छ। उत्तर नेपालीमा दिनुहोस्। केवल दिइएको कागजातको आधारमा उत्तर दिनुहोस्। तथ्य नबनाउनुहोस्। यदि उत्तर छैन भने "{NO_INFO}" भन्नुहोस्।

हालको मुद्दा: {case_id if case_id else "अज्ञात"}

प्रमाण (कागजातका अंशहरू):
{context}"""

        # ---- DYNAMIC TOKEN LIMIT (adjusted for Devanagari) ----
        if intent in ["LIST_CASES", "CASE_LOOKUP"]:
            token_limit = 450
        elif intent == "CASE_SUMMARY":
            token_limit = 1200
        else:
            token_limit = 800  # Raised from 600 for QA

        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": query.strip()},
                ],
                model=model_name,
                temperature=0.15,
                max_tokens=token_limit,
            )
            raw_answer = response.choices[0].message.content
            if raw_answer and raw_answer.strip():
                return clean_and_repair_nepali_output(raw_answer)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            # Fall through

    # ---- LLM failed or not available ----
    # If we have a header chunk, show its content as fallback
    for item in retrieved_items:
        if item.get("is_header", False):
            return f"उपलब्ध कागजातबाट निम्न जानकारी प्राप्त भएको छ:\n\n{item.get('content', '')}"
    return NO_INFO