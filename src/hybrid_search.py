import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from src.config import (
    CASE_INDEX_PATH,
    BM25_CANDIDATE_MULTIPLIER,
    VECTOR_CANDIDATE_CAP,
    QUERY_VEC_CACHE_SIZE,
    RRF_K,
)
from src.text_processor import char_ngram_tokenize, normalize_digits, clean_ocr_field

# ---- Alias mapping for common OCR/typo confusions ----
DECISION_ALIAS_MAP = {
    "1900": "9100",
    "1903": "9103",
    "1906": "9106",
    "1907": "9107",
    "1908": "9108",
}

_QUERY_VEC_CACHE = OrderedDict()

def encode_query(model, query: str) -> list[float]:
    cached = _QUERY_VEC_CACHE.get(query)
    if cached is not None:
        _QUERY_VEC_CACHE.move_to_end(query)
        return cached
    vector = model.encode([query], normalize_embeddings=True)[0].tolist()
    _QUERY_VEC_CACHE[query] = vector
    while len(_QUERY_VEC_CACHE) > QUERY_VEC_CACHE_SIZE:
        _QUERY_VEC_CACHE.popitem(last=False)
    return vector

_CASE_INDEX = None
_LOOKUP = {"n": -1, "id": None, "case": {}, "decision": {}, "headers": {}}

def get_case_index():
    global _CASE_INDEX
    if _CASE_INDEX is None:
        try:
            with open(CASE_INDEX_PATH, "r", encoding="utf-8") as f:
                _CASE_INDEX = json.load(f)
        except Exception:
            _CASE_INDEX = {}
    return _CASE_INDEX

def reset_lookup_indexes():
    global _CASE_INDEX
    _CASE_INDEX = None
    _LOOKUP["n"] = -1
    _LOOKUP["id"] = None
    _LOOKUP["case"] = {}
    _LOOKUP["decision"] = {}
    _LOOKUP["headers"] = {}

def get_lookup_indexes(chunk_metadata: list) -> dict:
    n = len(chunk_metadata)
    meta_id = id(chunk_metadata)
    if _LOOKUP["n"] == n and _LOOKUP["id"] == meta_id:
        return _LOOKUP
    case_map = defaultdict(list)
    decision_map = defaultdict(list)
    headers = {}
    for i, meta in enumerate(chunk_metadata):
        cid = meta.get("case_id") or ""
        if cid:
            case_map[cid].append(i)
            if meta.get("is_header"):
                headers[cid] = i
        dec = normalize_digits(str(meta.get("decision_no") or ""))
        if dec:
            decision_map[dec].append(i)
    _LOOKUP["n"] = n
    _LOOKUP["id"] = meta_id
    _LOOKUP["case"] = dict(case_map)
    _LOOKUP["decision"] = dict(decision_map)
    _LOOKUP["headers"] = headers
    return _LOOKUP

def chunks_for_decision(chunk_metadata: list, decision_no: str, case_id: str | None = None) -> list:
    lookups = get_lookup_indexes(chunk_metadata)
    indices = []
    if case_id:
        indices = lookups["case"].get(case_id, [])
    if not indices and decision_no:
        indices = lookups["decision"].get(normalize_digits(str(decision_no)), [])
    return [chunk_metadata[i] for i in indices]

def _top_n_from_scores(scores, n: int, allowed_indices=None) -> list[int]:
    if n <= 0:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    if allowed_indices is not None:
        allowed = np.fromiter(allowed_indices, dtype=np.int64)
        if allowed.size == 0:
            return []
        subset = arr[allowed]
        k = min(n, allowed.size)
        if k == allowed.size:
            order = np.argsort(subset)[::-1]
            return allowed[order].tolist()
        part = np.argpartition(subset, -k)[-k:]
        order = part[np.argsort(subset[part])[::-1]]
        return allowed[order].tolist()
    total = arr.size
    if total == 0:
        return []
    k = min(n, total)
    if k == total:
        return np.argsort(arr)[::-1].tolist()
    part = np.argpartition(arr, -k)[-k:]
    return part[np.argsort(arr[part])[::-1]].tolist()

SUMMARY_TERMS = [
    "summary", "summarize", "सारांश", "विस्तृत जानकारी",
    "संक्षिप्त विवरण", "पूरा विवरण",
]
ABOUT_TERMS = [
    "के सम्बन्धी", "के बारे", "विषय के", "मुद्दाको प्रकार",
    "what is this case about", "case about",
]
PRINCIPLE_TERMS = ["कानूनी सिद्धान्त", "मुख्य सिद्धान्त", "ratio", "प्रतिपादन गरिएको", "नजिर"]
COMPARISON_TERMS = ["compare", "comparison", "difference", "फरक", "तुलना", "दुवै मुद्दा", "यी दुई", "दुबै"]
RELATIVE_PRONOUNS = ["यस", "उक्त", "त्यस", "यो", "this", "said", "above", "सो"]

LIST_CUES = [
    "कुन-कुन", "कुन कुन", "के-के", "के के",
    "कति वटा", "कति निर्णय", "कति मुद्दा",
    "सबै मुद्दा", "सबै निर्णय", "सबै फैसला",
    "सूची", "लिस्ट", "फैसलाहरू",
]

def detect_query_intent(query: str) -> str:
    q = (query or "").lower().strip()
    normalized = normalize_digits(q)
    numbers = re.findall(r"\b([0-9]{3,4})\b", normalized)
    is_section_number = any(f"धारा {num}" in normalized or f"दफा {num}" in normalized or f"नियम {num}" in normalized for num in numbers)
    if any(term in q for term in COMPARISON_TERMS) or (len(numbers) >= 2 and not is_section_number):
        return "COMPARISON"
    if any(term in q for term in PRINCIPLE_TERMS) or ("सिद्धान्त" in q and "कानून" in q):
        return "LEGAL_PRINCIPLE"
    if any(kw in q for kw in ["न्यायाधीश", "इजलास", "बेन्च", "न्यादिश", "न्यादिष"]):
        return "FACTUAL"
    if any(kw in q for kw in ["निवेदक", "पुनरावेदक", "विपक्षी", "प्रत्यर्थी", "पक्षकार", "कानून व्यवसायी", "अधिवक्ता", "वकील"]):
        return "FACTUAL"
    if "मिति" in q or "कहिले" in q or "जरीवाना" in q or "जरिवाना" in q or "सजाय" in q:
        return "FACTUAL"
    if ("अन्तिम" in q and "आदेश" in q) or "खारेज" in q or "सदर" in q or "उल्टी" in q or "सफाइ" in q:
        return "FACTUAL"
    if any(term in q for term in ABOUT_TERMS):
        return "CASE_ABOUT"
    if any(term in q for term in SUMMARY_TERMS) or "मुद्दा के थियो" in q or "फैसला के थियो" in q:
        return "CASE_SUMMARY"
    has_case = any(w in q for w in ["मुद्दा", "मुद्धा", "केस", "फैसला", "निर्णय", "निर्णयहरू", "नजिर", "कागजात"])
    if has_case and any(cue in q for cue in LIST_CUES):
        return "LIST_CASES"
    if "pdf" in q or "source" in q or "कुन document" in q or "कुन कागजात" in q:
        return "CASE_LOOKUP"
    if re.search(r"(?:निर्णय\s*नं\.?|decision\s*(?:no|number)|नं\.)\s*[०-९0-9]+", q, re.I):
        return "CASE_LOOKUP"
    if any(x in q for x in ["section", "दफा", "धारा", "कानून", "ऐन", "नियम", "नियमावली"]):
        return "LEGAL_PROVISION"
    return "LEGAL_QA"

def decision_exists(decision_no: str, chunk_metadata: list | None = None) -> bool:
    dec = normalize_digits(str(decision_no or "")).strip()
    if not dec:
        return False
    # Apply alias mapping
    dec = DECISION_ALIAS_MAP.get(dec, dec)
    if dec in get_case_index():
        return True
    if chunk_metadata is not None:
        lookups = get_lookup_indexes(chunk_metadata)
        return bool(lookups["decision"].get(dec))
    return False

def snap_decision_number(decision_no: str, chunk_metadata: list | None = None, max_distance: int = 1) -> str:
    dec = normalize_digits(str(decision_no or "")).strip()
    if not dec:
        return dec
    # Apply alias mapping
    dec = DECISION_ALIAS_MAP.get(dec, dec)
    if decision_exists(dec, chunk_metadata):
        return dec

    known = set(get_case_index().keys())
    if chunk_metadata is not None:
        known |= set(get_lookup_indexes(chunk_metadata)["decision"].keys())
    if not known:
        return dec

    def _edit_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            prev = cur
        return prev[-1]

    best, best_dist = None, max_distance + 1
    for candidate in known:
        d = _edit_distance(dec, candidate)
        if d < best_dist:
            best, best_dist = candidate, d
    return best if best_dist <= max_distance else dec

def extract_query_identifiers(query: str, active_case_id: str = None) -> dict:
    q = query or ""
    normalized = normalize_digits(q)
    identifiers = {}

    # 1. Explicit decision number patterns
    m = re.search(r"(?:निर्णय\s*नं\.?|decision\s*(?:no|number)|नि\.नं\.?)\s*[:.-]?\s*([0-9]{3,})", normalized, re.I)
    if m:
        identifiers["decision_no"] = m.group(1)
    else:
        # Fallback: "निर्णय 9106" without नं
        m = re.search(r"निर्णय\s*([0-9]{3,})", normalized, re.I)
        if m:
            identifiers["decision_no"] = m.group(1)

    # 2. If still nothing, look for standalone numbers
    #    but ONLY if query contains case keywords AND the number is not a law year/section
    if not identifiers.get("decision_no"):
        case_keywords = ["निर्णय", "मुद्दा", "फैसला", "case", "decision", "नं", "नं."]
        if any(kw in q.lower() for kw in case_keywords):
            # Exclude numbers preceded by ऐन, दफा, धारा, नियम (law years, sections)
            # Also exclude numbers that are part of a date (e.g., २०६३-०४-०५)
            candidates = re.findall(
                r"(?<!ऐन\s)(?<!दफा\s)(?<!धारा\s)(?<!नियम\s)(?<![./-])\b([0-9]{4})\b(?![./-])",
                normalized
            )
            if len(candidates) == 1:
                identifiers["decision_no"] = candidates[0]
                identifiers["_inferred_as_standalone"] = True

    # 3. Apply alias mapping
    if identifiers.get("decision_no"):
        dec = identifiers["decision_no"]
        if dec in DECISION_ALIAS_MAP:
            identifiers["decision_no"] = DECISION_ALIAS_MAP[dec]
            identifiers["_alias_used"] = True

    # 4. Pronoun reference to active session case
    if not identifiers.get("decision_no") and active_case_id:
        if any(pronoun in q.lower() for pronoun in RELATIVE_PRONOUNS):
            identifiers["decision_no"] = active_case_id.replace("decision_", "")
            identifiers["_inferred_from_context"] = True

    # 5. Source filename
    m = re.search(r"(nkp[_\-][0-9]+(?:[_\-][0-9]+)?(?:[_\-]part[0-9]+)?\.pdf)", q, re.I)
    if m:
        identifiers["source"] = m.group(1)

    # 6. Multiple decision numbers for comparison
    all_numbers = re.findall(r"(?<!ऐन\s)(?<!दफा\s)(?<!धारा\s)(?<!नियम\s)\b([0-9]{4})\b", normalized)
    unique_numbers = list(dict.fromkeys(all_numbers))
    if len(unique_numbers) >= 2:
        identifiers["multiple_decision_nos"] = unique_numbers

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

def _as_bool(val) -> bool:
    return val in (True, "True", "true", 1, "1")

def _make_result(meta: dict, score: float, extra: dict = None) -> dict:
    parties = meta.get("parties", {})
    if isinstance(parties, dict):
        parties = {k: clean_ocr_field(v) for k, v in parties.items()}
    return {
        "score": score,
        "vector_score": extra.get("vector_score", 0.0) if extra else 0.0,
        "bm25_score": extra.get("bm25_score", 0.0) if extra else 0.0,
        "lexical_score": extra.get("lexical_score", 0.0) if extra else 0.0,
        "source": meta.get("source"),
        "page": meta.get("page"),
        "total_pages": meta.get("total_pages"),
        "content": meta.get("content", ""),
        "decision_no": meta.get("decision_no", ""),
        "decision_no_original": meta.get("decision_no_original", ""),
        "date": clean_ocr_field(meta.get("date", "")),
        "subject": clean_ocr_field(meta.get("subject", "")),
        "case_id": meta.get("case_id", ""),
        "parties": parties,
        "case_type": clean_ocr_field(meta.get("case_type", "")),
        "judges": clean_ocr_field(meta.get("judges", "")),
        "appellant_lawyer": clean_ocr_field(meta.get("appellant_lawyer", "")),
        "respondent_lawyer": clean_ocr_field(meta.get("respondent_lawyer", "")),
        "provisions": clean_ocr_field(meta.get("provisions", "")),
        "final_order": clean_ocr_field(meta.get("final_order", "")),
        "legal_principle": clean_ocr_field(meta.get("legal_principle", "")),
        "precedents": clean_ocr_field(meta.get("precedents", "")),
        "prakaran_no": meta.get("prakaran_no"),
        "is_header": _as_bool(meta.get("is_header", False)),
    }

def perform_hybrid_search(
    query: str,
    collection,
    model,
    bm25,
    chunk_metadata: list,
    top_k: int = 5,
    alpha: float = 0.5,
    current_case: dict = None,
    identifiers: dict = None,
) -> list[dict]:
    if identifiers is None:
        identifiers = extract_query_identifiers(query)
    total = len(chunk_metadata)
    if total == 0:
        return []
    lookups = get_lookup_indexes(chunk_metadata)
    target_case_id = None
    target_decision_no = None
    candidate_indices = None
    candidate_set = None

    # Resolve target case
    if identifiers.get("decision_no"):
        target_decision_no = identifiers["decision_no"]
        case_index = get_case_index()
        case_info = case_index.get(target_decision_no)
        if case_info:
            target_case_id = case_info["case_id"]
        else:
            dec_hits = lookups["decision"].get(target_decision_no, [])
            if dec_hits:
                target_case_id = chunk_metadata[dec_hits[0]].get("case_id")
    if current_case and current_case.get("case_id"):
        target_case_id = current_case["case_id"]
        if not identifiers.get("decision_no") and target_case_id.startswith("decision_"):
            target_decision_no = target_case_id.replace("decision_", "")

    if target_case_id and target_case_id in lookups["case"]:
        candidate_indices = lookups["case"][target_case_id]
        candidate_set = set(candidate_indices)
    elif target_decision_no and target_decision_no in lookups["decision"]:
        candidate_indices = lookups["decision"][target_decision_no]
        candidate_set = set(candidate_indices)
        target_case_id = chunk_metadata[candidate_indices[0]].get("case_id") or target_case_id
    else:
        candidate_indices = None
        candidate_set = None

    # Detect if query asks about final outcome (to boost late pages)
    final_keywords = ["अन्तिम", "बदर", "उल्टी", "सदर", "फैसला के", "ठहर", "त्रुटिपूर्ण", "किन"]
    query_is_about_final = any(kw in query for kw in final_keywords) and (
        "आदेश" in query or "निर्णय" in query or "ठहर" in query
    )

    # Vector search
    vector = encode_query(model, query)
    corpus_size = collection.count()
    search_n = min(max(top_k * BM25_CANDIDATE_MULTIPLIER, 30), total if candidate_indices is None else len(candidate_indices))
    safe_n = max(1, min(search_n * 2, VECTOR_CANDIDATE_CAP, corpus_size))
    query_kwargs = {
        "query_embeddings": [vector],
        "n_results": safe_n,
        "include": ["distances", "metadatas"],
    }
    if target_case_id:
        query_kwargs["where"] = {"case_id": target_case_id}
    try:
        vec_res = collection.query(**query_kwargs)
    except Exception:
        query_kwargs.pop("where", None)
        vec_res = collection.query(**query_kwargs)

    vec_scores = {}
    if vec_res.get("ids") and vec_res["ids"]:
        for idx_id, dist, meta in zip(
            vec_res["ids"][0], vec_res["distances"][0], vec_res["metadatas"][0]
        ):
            idx = meta.get("chunk_index")
            if idx is None:
                try:
                    idx = int(str(idx_id).replace("doc_chunk_", ""))
                except Exception:
                    continue
            else:
                idx = int(idx)
            if candidate_set is not None and idx not in candidate_set:
                continue
            if 0 <= idx < total:
                vec_scores[idx] = 1.0 / (1.0 + float(dist))

    # BM25
    query_tokens = char_ngram_tokenize(query)
    bm25_scores = {}
    if hasattr(bm25, "search"):
        bm25_top, sparse_scores = bm25.search(query_tokens, top_k=search_n, allowed_indices=candidate_indices)
        for doc_idx in bm25_top:
            bm25_scores[doc_idx] = float(sparse_scores.get(doc_idx, 0.0))
    else:
        bm25_scores_raw = bm25.get_scores(query_tokens)
        bm25_top = _top_n_from_scores(bm25_scores_raw, search_n, candidate_indices)
        for doc_idx in bm25_top:
            bm25_scores[doc_idx] = float(bm25_scores_raw[doc_idx])

    # RRF
    vec_ranks = {doc: rank for rank, doc in enumerate(sorted(vec_scores.keys(), key=lambda x: vec_scores[x], reverse=True), start=1)}
    bm25_ranks = {doc: rank for rank, doc in enumerate(sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True), start=1)}
    all_candidates = set(vec_scores) | set(bm25_scores)
    if candidate_indices and not all_candidates:
        all_candidates = set(candidate_indices[:top_k * 2])
    if target_case_id:
        header_idx = lookups["headers"].get(target_case_id)
        if header_idx is not None:
            all_candidates.add(header_idx)

    # For final‑outcome queries, explicitly add the last pages of the target case
    if query_is_about_final and target_case_id and candidate_indices:
        # Get all chunk indices for this case
        case_chunks = candidate_indices
        # Sort by page descending, take top 3 (last pages)
        last_page_indices = sorted(case_chunks, key=lambda i: chunk_metadata[i].get("page", 0), reverse=True)[:3]
        for idx in last_page_indices:
            all_candidates.add(idx)

    # Score each candidate
    scored = []
    for doc in all_candidates:
        v_rank = vec_ranks.get(doc, 9999)
        b_rank = bm25_ranks.get(doc, 9999)
        rrf = (1.0 / (RRF_K + v_rank)) + (1.0 / (RRF_K + b_rank))
        v_score = vec_scores.get(doc, 0.0)
        b_score = bm25_scores.get(doc, 0.0)
        fused = rrf * 2.0 + 0.1 * v_score + 0.1 * b_score
        meta = chunk_metadata[doc]
        # Exact decision match boost
        if target_decision_no and normalize_digits(str(meta.get("decision_no", ""))) == target_decision_no:
            fused += 0.5
        # Header boost
        if meta.get("is_header"):
            fused += 0.3
        # Final‑reasoning boost: if query is about final outcome and chunk is near the end
        if query_is_about_final:
            page = meta.get("page", 0)
            total_pages = meta.get("total_pages", 0)
            if total_pages > 0 and page > total_pages * 0.7:
                content = meta.get("content", "")
                if any(k in content for k in ["तसर्थ", "अतः", "यसरी", "बदर", "उल्टी", "सदर", "ठहर"]):
                    fused += 0.5
        res = _make_result(meta, fused, {"vector_score": v_score, "bm25_score": b_score})
        res["index"] = doc
        scored.append(res)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Neighbour expansion
    expanded = []
    seen = set()
    for res in scored[:top_k]:
        idx = res["index"]
        if idx not in seen:
            expanded.append(res)
            seen.add(idx)
        for offset in (-1, 1):
            n_idx = idx + offset
            if 0 <= n_idx < total and n_idx not in seen:
                n_meta = chunk_metadata[n_idx]
                if n_meta.get("case_id") == res["case_id"] and not n_meta.get("is_header"):
                    nr = _make_result(
                        n_meta,
                        res["score"] * 0.85,
                        {"vector_score": res["vector_score"] * 0.85, "bm25_score": res["bm25_score"] * 0.85}
                    )
                    nr["index"] = n_idx
                    expanded.append(nr)
                    seen.add(n_idx)

    expanded.sort(key=lambda x: x["score"], reverse=True)
    return expanded[:top_k]