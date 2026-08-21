import re
from collections import defaultdict
from src.text_processor import char_ngram_tokenize, normalize_digits

SUMMARY_TERMS = [
    "summary", "summarize", "what was this case about", "case about",
    "सारांश", "सार", "मुद्दा के थियो", "मुद्दा के हो", "फैसला के थियो",
    "यो मुद्दा", "के सम्बन्धी", "विस्तृत जानकारी"
]

RELATIVE_PRONOUNS = ["यस", "उक्त", "त्यस", "यो", "this", "said", "above", "सो"]

def detect_query_intent(query: str) -> str:
    q = (query or "").lower().strip()
    list_phrases = [
        "कुन कुन मुद्दा", "कुन-कुन मुद्दा", "कुन मुद्दा", "कस्ता मुद्दा",
        "मुद्दाको जानकारी", "के के मुद्दा", "कुन-कुन केस",
        "list of cases", "what cases", "available cases", "कुन-कुन मुद्दाको जानकारी",
        "what information do you have", "which cases", "cases information", "tell me about cases",
        "कुन-कुन मुद्दा छन्", "के के मुद्दा छन्"
    ]
    if any(phrase in q for phrase in list_phrases):
        return "LIST_CASES"
    if any(term in q for term in SUMMARY_TERMS):
        return "CASE_SUMMARY"
    if re.search(r"(?:निर्णय\s*नं\.?|decision\s*(?:no|number)|नं\.)\s*[०-९0-9]+", q, re.I):
        return "CASE_LOOKUP"
    if re.search(r"nkp[_\s-]*[०-९0-9]+", q, re.I) or q.endswith(".pdf") or ".pdf" in q:
        return "CASE_LOOKUP"
    if any(x in q for x in ["section", "दफा", "धारा", "कानून", "ऐन", "नियम"]):
        return "LEGAL_PROVISION"
    if any(x in q for x in ["compare", "difference", "फरक", "तुलना"]):
        return "COMPARISON"
    return "LEGAL_QA"

def extract_query_identifiers(query: str, active_case_id: str = None) -> dict:
    q = query or ""
    normalized = normalize_digits(q)
    identifiers = {}
    
    # ---- 1. Explicit Nepali decision number patterns ----
    m = re.search(r"(?:निर्णय\s*नं\.?|decision\s*(?:no|number)|नं\.)\s*([0-9]{3,})", normalized, re.I)
    if m:
        identifiers["decision_no"] = m.group(1)
    else:
        # Simpler: number after "निर्णय"
        m = re.search(r"निर्णय\s*([0-9]{3,})", normalized, re.I)
        if m:
            identifiers["decision_no"] = m.group(1)
        else:
            m = re.search(r"(?:निर्णय|नं\.)\s*([0-9]{4})", normalized, re.I)
            if m:
                identifiers["decision_no"] = m.group(1)

    # ---- 2. English patterns ----
    if not identifiers.get("decision_no"):
        m = re.search(r"(?:case\s*(?:no|number)|cases\s*no)\s*[:. ]?\s*([0-9]{3,})", normalized, re.I)
        if m:
            identifiers["decision_no"] = m.group(1)
    if not identifiers.get("decision_no"):
        m = re.search(r"decision\s*(?:no|number)\s*[:. ]?\s*([0-9]{3,})", normalized, re.I)
        if m:
            identifiers["decision_no"] = m.group(1)

    # ---- 3. Standalone number if the query is clearly about a case ----
    if not identifiers.get("decision_no"):
        # Check if the query contains case-related keywords and a 3-4 digit number
        case_keywords = ["निर्णय", "मुद्दा", "फैसला", "case", "decision", "नं"]
        if any(kw in q.lower() for kw in case_keywords):
            standalone_numbers = re.findall(r"\b([0-9]{3,4})\b", normalized)
            if len(standalone_numbers) == 1:
                identifiers["decision_no"] = standalone_numbers[0]
                identifiers["_inferred_as_standalone"] = True

    # ---- 4. Relative pronoun fallback ----
    if not identifiers.get("decision_no") and active_case_id:
        if any(pronoun in q.lower() for pronoun in RELATIVE_PRONOUNS):
            identifiers["decision_no"] = active_case_id.replace("decision_", "")
            identifiers["_inferred_from_context"] = True

    # ---- Source patterns (for PDF filenames) ----
    m = re.search(r"(nkp[_\-][0-9]+(?:[_\-][0-9]+)?(?:[_\-]part[0-9]+)?\.pdf)", q, re.I)
    if m:
        identifiers["source"] = m.group(1)
    else:
        m = re.search(r"(nkp[_\-][0-9]+(?:[_\-][0-9]+)?(?:[_\-]part[0-9]+)?)", q, re.I)
        if m:
            identifiers["source_stem"] = m.group(1)
    return identifiers

def _normalize_for_match(s: str) -> str:
    s = normalize_digits(s or "").lower()
    s = re.sub(r"[^\w\u0900-\u097f]+", "", s, flags=re.UNICODE)
    return s

def _lexical_score(query: str, text: str) -> float:
    q = _normalize_for_match(query)
    t = _normalize_for_match(text)
    if not q or not t:
        return 0.0
    score = 0.0
    if q in t:
        score += 1.0
    q_tokens = set(char_ngram_tokenize(q))
    t_tokens = set(char_ngram_tokenize(t))
    if q_tokens and t_tokens:
        score += len(q_tokens & t_tokens) / max(len(q_tokens), 1)
    return min(score / 2.0, 1.0)

def _case_key(meta: dict) -> str:
    return str(meta.get("case_id") or meta.get("decision_no") or meta.get("source") or "unknown")

def perform_hybrid_search(query, collection, model, bm25, chunk_metadata, top_k=5, alpha=0.15, current_case=None):
    """
    Hybrid search with strict case filtering.
    If current_case is provided, we restrict to that case.
    """
    intent = detect_query_intent(query)
    # We need to extract identifiers here; but note that app.py already extracts them
    # with active_case_id. We'll just use the query again (or accept identifiers as parameter).
    # For simplicity, we'll re-extract, but we don't have active_case_id here.
    # So we'll call without active_case_id (the caller should have already set current_case).
    identifiers = extract_query_identifiers(query)
    total = len(chunk_metadata)
    if total == 0:
        return []

    # ---- RESTRICT TO CURRENT CASE ----
    candidate_indices = list(range(total))
    if current_case:
        case_id = current_case.get("case_id")
        if case_id:
            filtered = [i for i, m in enumerate(chunk_metadata) if m.get("case_id") == case_id]
            if filtered:
                candidate_indices = filtered
                # If we have a current case, we can also set the decision_no if not present
                if not identifiers.get("decision_no"):
                    # Extract from case_id
                    if case_id.startswith("decision_"):
                        identifiers["decision_no"] = case_id.replace("decision_", "")
            else:
                # No chunks for this case – return empty
                return []

    # ---- If explicit decision_no is present, filter strictly ----
    if identifiers.get("decision_no"):
        target = identifiers["decision_no"]
        filtered = []
        for i, m in enumerate(chunk_metadata):
            m_no = normalize_digits(str(m.get("decision_no", "")))
            if m_no == target:
                filtered.append(i)
        if filtered:
            candidate_indices = filtered
        else:
            # Try by case_id
            case_id = f"decision_{target}"
            filtered = [i for i, m in enumerate(chunk_metadata) if m.get("case_id") == case_id]
            if filtered:
                candidate_indices = filtered
            else:
                return []

    # ---- For summary/lookup, return all chunks for the case ----
    if intent in ("CASE_SUMMARY", "CASE_LOOKUP"):
        all_case_chunks = []
        for idx, meta in enumerate(chunk_metadata):
            if idx in candidate_indices:
                all_case_chunks.append((idx, meta))
        all_case_chunks.sort(key=lambda x: (x[1].get("page", 0), x[1].get("index", 0)))
        results = []
        for idx, meta in all_case_chunks[:50]:
            results.append({
                "index": idx,
                "score": 1.0,
                "vector_score": 1.0,
                "bm25_score": 1.0,
                "lexical_score": 1.0,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "total_pages": meta.get("total_pages"),
                "content": meta.get("content", ""),
                "page_text": meta.get("page_text", meta.get("content", "")),
                "decision_no": meta.get("decision_no", ""),
                "date": meta.get("date", ""),
                "subject": meta.get("subject", ""),
                "case_id": meta.get("case_id", ""),
                "parties": meta.get("parties", {}),
                "case_type": meta.get("case_type", ""),
                "judges": meta.get("judges", ""),
                "appellant_lawyer": meta.get("appellant_lawyer", ""),
                "respondent_lawyer": meta.get("respondent_lawyer", ""),
                "provisions": meta.get("provisions", ""),
                "prakaran_no": meta.get("prakaran_no"),
                "is_header": meta.get("is_header", False),
            })
        results.sort(key=lambda x: (x.get("is_header", False), x.get("page", 999)), reverse=True)
        return results

    # ---- Normal hybrid search on candidate_indices ----
    search_n = min(max(top_k * 8, 20), len(candidate_indices))
    vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    vec_res = collection.query(
        query_embeddings=[vector],
        n_results=min(search_n * 3, 100),
        include=["distances", "metadatas", "documents"]
    )

    vec_scores = {}
    if vec_res.get("ids"):
        for idx_id, dist, meta, doc in zip(
            vec_res["ids"][0], vec_res["distances"][0], vec_res["metadatas"][0], vec_res["documents"][0]
        ):
            try:
                idx = int(idx_id.replace("doc_chunk_", ""))
            except Exception:
                continue
            if idx not in candidate_indices:
                continue
            vec_scores[idx] = 1.0 / (1.0 + float(dist))

    bm25_scores_raw = bm25.get_scores(char_ngram_tokenize(query))
    candidate_scores = {i: bm25_scores_raw[i] for i in candidate_indices}
    max_bm25 = max(candidate_scores.values()) if candidate_scores else 1.0
    if max_bm25 <= 0:
        max_bm25 = 1.0

    results = []
    union = set(vec_scores) | set(candidate_scores.keys())
    for i in union:
        meta = chunk_metadata[i]
        vector_score = vec_scores.get(i, 0.0)
        bm_score = max(0.0, candidate_scores.get(i, 0.0)) / max_bm25
        lexical = _lexical_score(query, meta.get("content", ""))
        fused = alpha * vector_score + (1 - alpha) * bm_score
        fused += 0.20 * lexical
        if intent in ("CASE_SUMMARY", "CASE_LOOKUP") and meta.get("is_header", False):
            fused += 2.0

        results.append({
            "index": i,
            "score": fused,
            "vector_score": vector_score,
            "bm25_score": bm_score,
            "lexical_score": lexical,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "total_pages": meta.get("total_pages"),
            "content": meta.get("content", ""),
            "page_text": meta.get("page_text", meta.get("content", "")),
            "decision_no": meta.get("decision_no", ""),
            "date": meta.get("date", ""),
            "subject": meta.get("subject", ""),
            "case_id": meta.get("case_id", ""),
            "parties": meta.get("parties", {}),
            "case_type": meta.get("case_type", ""),
            "judges": meta.get("judges", ""),
            "appellant_lawyer": meta.get("appellant_lawyer", ""),
            "respondent_lawyer": meta.get("respondent_lawyer", ""),
            "provisions": meta.get("provisions", ""),
            "prakaran_no": meta.get("prakaran_no"),
            "is_header": meta.get("is_header", False),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # ---- Dominant case inference (only if no current_case and no decision_no) ----
    if not current_case and not identifiers.get("decision_no") and intent != "LIST_CASES":
        top_case_counter = defaultdict(int)
        for res in results[:max(top_k, 10)]:
            case_id = res.get("case_id")
            if case_id:
                top_case_counter[case_id] += 1
        if top_case_counter:
            dominant_case = max(top_case_counter, key=top_case_counter.get)
            if top_case_counter[dominant_case] >= max(top_k, 10) / 2:
                filtered_results = [r for r in results if r.get("case_id") == dominant_case]
                if filtered_results:
                    return filtered_results[:top_k]

    # ---- Neighbouring chunk expansion ----
    if intent not in ("CASE_SUMMARY", "CASE_LOOKUP", "LIST_CASES"):
        expanded = []
        seen_indices = set()
        for res in results[:top_k]:
            idx = res["index"]
            if idx not in seen_indices:
                expanded.append(res)
                seen_indices.add(idx)
            prev_idx = idx - 1
            if prev_idx >= 0 and prev_idx not in seen_indices:
                prev_meta = chunk_metadata[prev_idx]
                if prev_meta.get("case_id") == res["case_id"]:
                    prev_res = {
                        "index": prev_idx,
                        "score": res["score"] * 0.8,
                        "vector_score": res["vector_score"] * 0.8,
                        "bm25_score": res["bm25_score"] * 0.8,
                        "lexical_score": res["lexical_score"] * 0.8,
                        "source": prev_meta.get("source"),
                        "page": prev_meta.get("page"),
                        "total_pages": prev_meta.get("total_pages"),
                        "content": prev_meta.get("content", ""),
                        "page_text": prev_meta.get("page_text", prev_meta.get("content", "")),
                        "decision_no": prev_meta.get("decision_no", ""),
                        "date": prev_meta.get("date", ""),
                        "subject": prev_meta.get("subject", ""),
                        "case_id": prev_meta.get("case_id", ""),
                        "parties": prev_meta.get("parties", {}),
                        "case_type": prev_meta.get("case_type", ""),
                        "judges": prev_meta.get("judges", ""),
                        "appellant_lawyer": prev_meta.get("appellant_lawyer", ""),
                        "respondent_lawyer": prev_meta.get("respondent_lawyer", ""),
                        "provisions": prev_meta.get("provisions", ""),
                        "prakaran_no": prev_meta.get("prakaran_no"),
                        "is_header": prev_meta.get("is_header", False),
                    }
                    expanded.append(prev_res)
                    seen_indices.add(prev_idx)
            next_idx = idx + 1
            if next_idx < total and next_idx not in seen_indices:
                next_meta = chunk_metadata[next_idx]
                if next_meta.get("case_id") == res["case_id"]:
                    next_res = {
                        "index": next_idx,
                        "score": res["score"] * 0.8,
                        "vector_score": res["vector_score"] * 0.8,
                        "bm25_score": res["bm25_score"] * 0.8,
                        "lexical_score": res["lexical_score"] * 0.8,
                        "source": next_meta.get("source"),
                        "page": next_meta.get("page"),
                        "total_pages": next_meta.get("total_pages"),
                        "content": next_meta.get("content", ""),
                        "page_text": next_meta.get("page_text", next_meta.get("content", "")),
                        "decision_no": next_meta.get("decision_no", ""),
                        "date": next_meta.get("date", ""),
                        "subject": next_meta.get("subject", ""),
                        "case_id": next_meta.get("case_id", ""),
                        "parties": next_meta.get("parties", {}),
                        "case_type": next_meta.get("case_type", ""),
                        "judges": next_meta.get("judges", ""),
                        "appellant_lawyer": next_meta.get("appellant_lawyer", ""),
                        "respondent_lawyer": next_meta.get("respondent_lawyer", ""),
                        "provisions": next_meta.get("provisions", ""),
                        "prakaran_no": next_meta.get("prakaran_no"),
                        "is_header": next_meta.get("is_header", False),
                    }
                    expanded.append(next_res)
                    seen_indices.add(next_idx)
        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded[:top_k * 3]

    return results[:top_k if not (identifiers.get("decision_no") and intent in {"CASE_SUMMARY", "CASE_LOOKUP"}) else min(30, max(top_k, 10))]