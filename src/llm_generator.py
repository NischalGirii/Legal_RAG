import os
from groq import Groq
from dotenv import load_dotenv
from src.text_processor import clean_and_repair_nepali_output

load_dotenv()


def _format_case_header(item: dict) -> str:
    parties = item.get("parties") or {}
    return (
        f"CASE_ID: {item.get('case_id', 'UNKNOWN')}\n"
        f"निर्णय नं.: {item.get('decision_no', 'अज्ञात')}\n"
        f"फैसला मिति: {item.get('date', 'अज्ञात')}\n"
        f"विषय: {item.get('subject', 'अज्ञात')}\n"
        f"अदालत: सर्वोच्च अदालत\n"
        f"पुनरावेदक/विपक्षी: {parties.get('appellant', 'अज्ञात')}\n"
        f"प्रत्यर्थी/निवेदक: {parties.get('respondent', 'अज्ञात')}"
    )


def generate_nepali_answer(query: str, retrieved_items: list, model_name: str = "openai/gpt-oss-20b") -> str:
    if not retrieved_items:
        return "उपलब्ध कानूनी कागजातमा तपाईंको प्रश्नसँग सान्दर्भिक प्रमाण फेला परेन।"

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Group evidence by case so facts from separate decisions cannot silently mix.
    grouped = {}
    for item in retrieved_items:
        case_id = item.get("case_id") or item.get("source") or "UNKNOWN"
        grouped.setdefault(case_id, []).append(item)

    evidence_blocks = []
    for case_id, items in grouped.items():
        first = items[0]
        page_seen = set()
        pages = []
        for item in sorted(items, key=lambda x: (x.get("page", 0), x.get("index", 0))):
            page = item.get("page")
            if page in page_seen:
                continue
            page_seen.add(page)
            pages.append(
                f"--- पृष्ठ {page} ---\n{item.get('page_text') or item.get('content', '')}"
            )
        evidence_blocks.append(
            "\n".join([_format_case_header(first), *pages])
        )

    context = "\n\n================ CASE BOUNDARY ================\n\n".join(evidence_blocks)
    is_summary = any(x in (query or "").lower() for x in [
        "summary", "summarize", "case about", "what was this case about",
        "सारांश", "सार", "मुद्दा के थियो", "मुद्दा के हो", "के सम्बन्धी"
    ])

    system_prompt = """
तपाईं नेपालको सर्वोच्च अदालतका फैसला तथा नेपाल कानुन पत्रिका (NKP) का कागजात विश्लेषण गर्ने नेपाली कानूनी सहायक हुनुहुन्छ।

कडा नियम:
1. केवल प्रदान गरिएको आधिकारिक कागजात-सन्दर्भबाट तथ्य निकाल्नुहोस्। बाहिरी तथ्य, अनुमान वा काल्पनिक कुरा नथप्नुहोस्।
2. USER QUERY मा कुनै निर्णय नं., PDF filename, case id वा स्पष्ट पक्षकार भएमा सोही मुद्दालाई मात्र मुख्य स्रोत बनाउनुहोस्। अर्को मुद्दाको तथ्य मिसाउन पाइँदैन।
3. प्रत्येक CASE BOUNDARY अलग कानूनी मुद्दा हो। एउटा case को तथ्य अर्को case मा सार्न हुँदैन।
4. कागजातमा कुरा पुष्टि नभए “उपलब्ध सन्दर्भबाट पुष्टि हुन सकेन” भन्नुहोस्। अनुमानलाई तथ्यको रूपमा प्रस्तुत नगर्नुहोस्।
5. OCR बाट बिग्रिएको अक्षर भए अर्थ सुरक्षित रहने गरी मात्र सफा गर्नुहोस्। कानूनी दफा, निर्णय नं., मिति र पक्षकारको नाम परिवर्तन नगर्नुहोस्।
6. उत्तर नेपाली देवनागरीमा, स्पष्ट र औपचारिक शैलीमा दिनुहोस्।
"""

    if is_summary:
        system_prompt += """
7. CASE SUMMARY प्रश्नमा निम्न संरचना प्रयोग गर्नुहोस्:
   **मुद्दाको परिचय**
   **मुख्य तथ्य**
   **मुख्य कानूनी प्रश्न**
   **अदालतको तर्क**
   **निर्णय/आदेश**
   **मुख्य कानूनी सिद्धान्त वा नजिर**
8. summary मा कागजातमा भेटिएको decision number, date, subject र parties लाई प्राथमिक रूपमा स्पष्ट गर्नुहोस्।
"""
    else:
        system_prompt += """
7. प्रश्नले मागेको कुराको मात्र सीधा उत्तर दिनुहोस्। सान्दर्भिक नभएको सम्पूर्ण case को विवरण नदोहर्याउनुहोस्।
"""

    user_prompt = f"""
OFFICIAL LEGAL EVIDENCE:
{context}

USER QUESTION:
{query}

उत्तर तयार गर्दा माथिका CASE BOUNDARY हरूको सम्मान गर्नुहोस्।
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            model=model_name,
            temperature=0.15,
            max_tokens=1800 if is_summary else 1200,
        )
        raw_answer = response.choices[0].message.content
        return clean_and_repair_nepali_output(raw_answer)
    except Exception as e:
        return f"LLM Generation Error: {str(e)}"