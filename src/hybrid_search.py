import re
from collections import defaultdict
from src.text_processor import char_ngram_tokenize, normalize_digits

SUMMARY_TERMS = [
    "summary", "summarize", "what was this case about", "case about",
    "सारांश", "सार", "मुद्दा के थियो", "मुद्दा के हो", "फैसला के थियो",
    "यो मुद्दा", "के सम्बन्धी", "विस्तृत जानकारी"
]

def detect_query_intent(query: str) -> str:
    q = (query or "").lower().strip()
    # Check for list cases intent first
    list_phrases = [
        "कुन कुन मुद्दा", "कुन-कुन मुद्दा", "कुन मुद्दा", "कस्ता मुद्दा",
        "मुद्दाको जानकारी", "के के मुद्दा", "कुन-कुन केस",
        "list of cases", "what cases", "available cases", "कुन-कुन मुद्दाको जानकारी"
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

def extract_query_identifiers(query: str) -> dict:
    q = query or ""
    normalized = normalize_digits(q)
    identifiers = {}
    # Strict pattern: must have निर्णय/नं. etc.
    m = re.search(r"(?:निर्णय\s*नं\.?|decision\s*(?:no|number)|नं\.)\s*([0-9]{3,})", normalized, re.I)
    if m:
        identifiers["decision_no"] = m.group(1)
    else:
        # Simpler: any number after "निर्णय"
        m = re.search(r"निर्णय\s*([0-9]{3,})", normalized, re.I)
        if m:
            identifiers["decision_no"] = m.group(1)
        else:
            # Fallback: any 4-digit number near "निर्णय" or "नं."
            m = re.search(r"(?:निर्णय|नं\.)\s*([0-9]{4})", normalized, re.I)
            if m:
                identifiers["decision_no"] = m.group(1)

    # Source file patterns (unchanged)
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

def perform_hybrid_search(query, collection, model, bm25, chunk_metadata, top_k=5, alpha=0.15):
    """
    Hybrid search with default alpha=0.15 (15% vector, 85% BM25).
    """
    intent = detect_query_intent(query)
    identifiers = extract_query_identifiers(query)
    total = len(chunk_metadata)
    if total == 0:
        return []

    # ---- CASE SUMMARY WITH NO EXPLICIT DECISION NO ----
    if intent == "CASE_SUMMARY" and not identifiers.get("decision_no"):
        # Do a normal hybrid search to get top candidates
        candidate_indices = list(range(total))
        search_n = min(max(top_k * 10, 30), total)  # get more for inference
        vector = model.encode([query], normalize_embeddings=True)[0].tolist()
        vec_res = collection.query(
            query_embeddings=[vector],
            n_results=search_n,
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
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0

        # Build a temporary result set to identify the dominant case
        temp_results = []
        for i in candidate_indices:
            meta = chunk_metadata[i]
            vector_score = vec_scores.get(i, 0.0)
            bm_score = bm25_scores_raw[i] / max_bm25
            lexical = _lexical_score(query, meta.get("content", ""))
            fused = alpha * vector_score + (1 - alpha) * bm_score + 0.20 * lexical
            temp_results.append({
                "index": i,
                "score": fused,
                "case_id": meta.get("case_id"),
                "decision_no": meta.get("decision_no"),
            })
        temp_results.sort(key=lambda x: x["score"], reverse=True)

        # Count case occurrences in the top 15 results
        top_cases = [r["case_id"] for r in temp_results[:15] if r["case_id"]]
        if top_cases:
            case_counter = defaultdict(int)
            for c in top_cases:
                case_counter[c] += 1
            dominant_case = max(case_counter, key=case_counter.get)
            # If the dominant case appears at least 3 times (or >30% of top results)
            if case_counter[dominant_case] >= 3:
                # Fetch all chunks for that case
                all_case_chunks = []
                for i, meta in enumerate(chunk_metadata):
                    if meta.get("case_id") == dominant_case:
                        all_case_chunks.append((i, meta))
                if all_case_chunks:
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
                            "is_header": meta.get("is_header", False),
                        })
                    results.sort(key=lambda x: (x.get("is_header", False), x.get("page", 999)), reverse=True)
                    return results

    # ---- STRICT CASE FILTERING (if decision_no is present) ----
    if identifiers.get("decision_no"):
        target = identifiers["decision_no"]
        candidate_indices = []
        for i, m in enumerate(chunk_metadata):
            m_no = normalize_digits(str(m.get("decision_no", "")))
            if m_no == target:
                candidate_indices.append(i)
        if not candidate_indices:
            case_id = f"decision_{target}"
            for i, m in enumerate(chunk_metadata):
                if m.get("case_id") == case_id:
                    candidate_indices.append(i)
        if not candidate_indices:
            return []
        # For summary/lookup, fetch all chunks for this case
        if intent in ("CASE_SUMMARY", "CASE_LOOKUP"):
            all_case_chunks = []
            for i, meta in enumerate(chunk_metadata):
                if i in candidate_indices:  # all chunks already filtered
                    all_case_chunks.append((i, meta))
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
                    "is_header": meta.get("is_header", False),
                })
            results.sort(key=lambda x: (x.get("is_header", False), x.get("page", 999)), reverse=True)
            return results
        else:
            # Normal hybrid search within candidate_indices
            candidate_indices = candidate_indices
    else:
        candidate_indices = list(range(total))

    # ---- NORMAL HYBRID SEARCH (for other intents) ----
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

    # Build initial results
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
            "is_header": meta.get("is_header", False),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # ---- EXPAND WITH NEIGHBOURING CHUNKS (for non-summary queries) ----
    if intent not in ("CASE_SUMMARY", "CASE_LOOKUP", "LIST_CASES"):
        expanded = []
        seen_indices = set()
        for res in results[:top_k]:
            idx = res["index"]
            # Add the main chunk
            if idx not in seen_indices:
                expanded.append(res)
                seen_indices.add(idx)
            # Add previous chunk if it exists and belongs to the same case
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
                        "is_header": prev_meta.get("is_header", False),
                    }
                    expanded.append(prev_res)
                    seen_indices.add(prev_idx)
            # Add next chunk
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
                        "is_header": next_meta.get("is_header", False),
                    }
                    expanded.append(next_res)
                    seen_indices.add(next_idx)
        # Sort expanded by score descending, then limit to top_k * 3
        expanded.sort(key=lambda x: x["score"], reverse=True)
        return expanded[:top_k * 3]

    return results[:top_k if not (identifiers.get("decision_no") and intent in {"CASE_SUMMARY", "CASE_LOOKUP"}) else min(30, max(top_k, 10))]