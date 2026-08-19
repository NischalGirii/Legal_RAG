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

    m = re.search(r"(?:निर्णय\s*नं\.?|decision\s*(?:no|number)|नं\.)\s*([0-9]+)", normalized, re.I)
    if m:
        identifiers["decision_no"] = m.group(1)

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


def _meta_matches(item_meta: dict, identifiers: dict) -> bool:
    if identifiers.get("decision_no"):
        if normalize_digits(str(item_meta.get("decision_no", ""))) != identifiers["decision_no"]:
            return False
    if identifiers.get("source"):
        if item_meta.get("source", "").lower() != identifiers["source"].lower():
            return False
    if identifiers.get("source_stem"):
        if identifiers["source_stem"].lower() not in item_meta.get("source", "").lower():
            return False
    return True


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


def perform_hybrid_search(query, collection, model, bm25, chunk_metadata, top_k=5, alpha=0.8):
    """Case-aware hybrid retrieval with exact identifier filtering, vector/BM25 fusion and lexical reranking."""
    intent = detect_query_intent(query)
    identifiers = extract_query_identifiers(query)
    total = len(chunk_metadata)
    if total == 0:
        return []

    candidate_indices = list(range(total))
    exact_case = bool(identifiers)
    if exact_case:
        filtered = [
            i for i, m in enumerate(chunk_metadata)
            if _meta_matches(m, identifiers)
        ]
        if filtered:
            candidate_indices = filtered

    # Vector candidates. Retrieve more than needed so reranking has room.
    search_n = min(max(top_k * 8, 20), total)
    vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    vec_res = collection.query(
        query_embeddings=[vector],
        n_results=search_n,
        include=["distances", "metadatas", "documents"]
    )

    vec_scores = {}
    vec_meta = {}
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
            vec_meta[idx] = doc

    # BM25 scores across the corpus; only candidate indices are retained.
    bm25_scores_raw = bm25.get_scores(char_ngram_tokenize(query))
    ranked_bm25 = sorted(candidate_indices, key=lambda i: bm25_scores_raw[i], reverse=True)[:search_n]
    max_bm25 = max([float(bm25_scores_raw[i]) for i in ranked_bm25] or [1.0])
    if max_bm25 <= 0:
        max_bm25 = 1.0

    results = []
    union = set(vec_scores) | set(ranked_bm25)
    for i in union:
        meta = chunk_metadata[i]
        vector_score = vec_scores.get(i, 0.0)
        bm_score = max(0.0, float(bm25_scores_raw[i])) / max_bm25
        lexical = _lexical_score(query, meta.get("content", ""))
        identifier_boost = 1.0 if _meta_matches(meta, identifiers) else 0.0
        fused = alpha * vector_score + (1 - alpha) * bm_score
        # Exact case match dominates semantic similarity to prevent case mixing.
        if exact_case:
            fused += 0.65 * identifier_boost
        fused += 0.20 * lexical
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
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # For a case summary / explicit case lookup, return broader same-case evidence.
    if exact_case and intent in {"CASE_SUMMARY", "CASE_LOOKUP"} and results:
        target_case = _case_key(results[0])
        same_case = [
            {
                "index": i,
                "score": results[0]["score"] + 0.01,
                "vector_score": 0.0,
                "bm25_score": 0.0,
                "lexical_score": 0.0,
                "source": m.get("source"),
                "page": m.get("page"),
                "total_pages": m.get("total_pages"),
                "content": m.get("content", ""),
                "page_text": m.get("page_text", m.get("content", "")),
                "decision_no": m.get("decision_no", ""),
                "date": m.get("date", ""),
                "subject": m.get("subject", ""),
                "case_id": m.get("case_id", ""),
                "parties": m.get("parties", {}),
            }
            for i, m in enumerate(chunk_metadata)
            if _case_key(m) == target_case
        ]
        # Keep at most 2 chunks/page and up to 30 chunks for context efficiency.
        per_page = defaultdict(int)
        expanded = []
        for item in sorted(same_case, key=lambda x: (x["page"], x["index"])):
            if per_page[item["page"]] >= 2:
                continue
            expanded.append(item)
            per_page[item["page"]] += 1
            if len(expanded) >= 30:
                break
        if expanded:
            results = expanded

    return results[:top_k if not (exact_case and intent in {"CASE_SUMMARY", "CASE_LOOKUP"}) else min(30, max(top_k, 10))]