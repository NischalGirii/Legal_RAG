import re

def analyze_query(query_text: str) -> dict:
    """Classifies the query and extracts legal entities/sources."""
    analysis = {
        "raw_query": query_text,
        "source_filter": None,
        "entities": [],
        "query_type": "DIRECT_FACT",
        "requires_expansion": False
    }

    # 1. Source Detection (e.g., "nkp_2_2", "NKP 2/2")
    source_match = re.search(r'nkp[-_\s]?\d+[-_\s]?\d+', query_text, re.IGNORECASE)
    if source_match:
        # Normalize to nkp_X_Y format
        normalized = re.sub(r'[-_\s/]+', '_', source_match.group(0).lower())
        analysis["source_filter"] = normalized

    # 2. Entity & Role Detection
    if "निवेदक" in query_text or "विपक्षी" in query_text:
        analysis["query_type"] = "ENTITY_CASE_ANALYSIS"
        analysis["requires_expansion"] = True
    elif any(word in query_text for word in ["फैसला", "विवाद", "पृष्ठभूमि", "आदेश"]):
        analysis["query_type"] = "CASE_SUMMARY"
        analysis["requires_expansion"] = True
    elif "धारा" in query_text or "दफा" in query_text:
        analysis["query_type"] = "LEGAL_PROVISION_LOOKUP"
        
    return analysis