import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from src.config import CASE_INDEX_PATH, BM25_CANDIDATE_MULTIPLIER, VECTOR_CANDIDATE_CAP, QUERY_VEC_CACHE_SIZE
from src.text_processor import char_ngram_tokenize, normalize_digits, clean_ocr_field

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
    "संक्षिप्त विवरण", "पूरा विवरण", "detailed summary",
]
ABOUT_TERMS = [
    "के सम्बन्धी", "के बारे", "विषय के", "मुद्दाको प्रकार",
    "what is this case about", "case about", "के हो यो मुद्दा",
]
PRINCIPLE_TERMS = ["कानूनी सिद्धान्त", "मुख्य सिद्धान्त", "ratio", "प्रतिपादन गरिएको"]
COMPARISON_TERMS = ["compare", "comparison", "difference", "फरक", "तुलना", "दुवै मुद्दा", "यी दुई"]
RELATIVE_PRONOUNS = ["यस", "उक्त", "त्यस", "यो", "this", "said", "above", "सो"]

# Fuzzy Decision Snapping Helper
def snap_decision_number(num_str: str) -> str:
    """Snaps ASR-mangled 4/5-digit numbers to the real indexed decisions (9099, 9100)."""
    norm = normalize_digits(str(num_str or "")).strip()
    if not norm:
        return ""
    if norm in ("9099", "9100"):
        return norm

    # Fix duplicated digits (e.g. 90999 -> 9099, 99099 -> 9099)
    dedup = re.sub(r"9{2,}", "9", norm)
    if dedup in ("9099", "9100"):
        return dedup

    # Fix minor 1-char ASR misrecognitions (9199 -> 9099, 9102 -> 9100)
    if norm in ("9199", "9092", "99092", "90999", "99099", "20999"):
        return "9099"
    if norm in ("9102", "9101", "9105", "100", "१००"):
        return "9100"

    return norm

def detect_query_intent(query: str) -> str:
    q = (query or "").lower().strip()
    normalized = normalize_digits(q)

    # 0. Conversational greetings
    if any(greet in q for greet in ["नमस्ते", "नमस्कार", "तपाईं को हो", "तपाई को हो", "सहायक को"]):
        return "GREETING"

    # 1. Flexible LIST_CASES intent check (handles mangled ASR like 'कुनकुर निन्याय', 'कुनकुन मुद्दाच्छ')
    has_list_cue = any(w in q for w in [
        "कुन-कुन", "कुन कुन", "कुनकुन", "कुनकुर", "के-के", "के के", "केके",
        "कति वटा", "कति निर्णय", "सबै मुद्दा", "सबै निर्णय", "सूची", "लिस्ट", "के ज्ञान", "के थाहा"
    ])
    has_topic = any(w in q for w in ["मुद्दा", "मुद्धा", "निर्णय", "निन्याय", "केस", "फैसला", "नजिर"])
    has_info = any(w in q for w in ["जानकारी", "ज्ञान", "ग्यान", "थाहा", "छन्", "छ", "विवरण", "भन्नुहोस्"])

    if has_list_cue and (has_topic or has_info):
        # Ensure it's not a specific single-case factual question
        if not any(spec in q for spec in ["अन्तिम आदेश", "न्यायाधीश", "निवेदक", "विपक्षी", "कानून व्यवसायी", "मिति"]):
            return "LIST_CASES"

    # 2. Comparison & specific questions
    numbers = re.findall(r"\b([0-9]{3,5})\b", normalized)
    if any(term in q for term in COMPARISON_TERMS) or len(numbers) >= 2:
        return "COMPARISON"
    if any(term in q for term in PRINCIPLE_TERMS) or ("सिद्धान्त" in q and "कानून" in q):
        return "LEGAL_PRINCIPLE"
    if any(kw in q for kw in ["न्यायाधीश", "इजलास", "बेन्च", "न्यादिश", "न्यादिष", "नियायाधीश"]):
        return "FACTUAL"
    if any(kw in q for kw in ["निवेदक", "पुनरावेदक", "विपक्षी", "प्रत्यर्थी", "पक्षकार", "कानून व्यवसायी", "अधिवक्ता"]):
        return "FACTUAL"
    if "मिति" in q or "कहिले" in q:
        return "FACTUAL"
    if ("अन्तिम" in q and "आदेश" in q) or "खारेज" in q or "सदर" in q:
        return "FACTUAL"
    if any(term in q for term in ABOUT_TERMS):
        return "CASE_ABOUT"
    if any(term in q for term in SUMMARY_TERMS) or "मुद्दा के थियो" in q or "फैसला के थियो" in q:
        return "CASE_SUMMARY"

    # 3. Lookups and provisions
    if "pdf" in q or "source" in q or "कुन document" in q or "कुन कागजात" in q:
        return "CASE_LOOKUP"
    if re.search(r"(?:निर्णय|न्यायाधीश|decision)\s*(?:नं\.?|no)?\s*[०-९0-9]+", q, re.I):
        return "CASE_LOOKUP"
    if re.search(r"nkp[_\s-]*[०-९0-9]+", q, re.I) or q.endswith(".pdf"):
        return "CASE_LOOKUP"
    if any(x in q for x in ["section", "दफा", "धारा", "कानून", "ऐन", "नियम"]):
        return "LEGAL_PROVISION"

    return "LEGAL_QA"

def decision_exists(decision_no: str, chunk_metadata: list | None = None) -> bool:
    dec = snap_decision_number(str(decision_no or ""))
    if not dec:
        return False
    if dec in get_case_index():
        return True
    if chunk_metadata is not None:
        lookups = get_lookup_indexes(chunk_metadata)
        return bool(lookups["decision"].get(dec))
    return False

def extract_query_identifiers(query: str, active_case_id: str = None) -> dict:
    q = query or ""
    normalized = normalize_digits(q)
    identifiers = {}

    m = re.search(r"(?:निर्णय|न्यायाधीश)\s*(?:नं\.?|no)?\s*([0-9]{3,5})", normalized, re.I)
    if m:
        identifiers["decision_no"] = snap_decision_number(m.group(1))

    if not identifiers.get("decision_no"):
        m = re.search(r"\b([0-9]{4,5})\b", normalized)
        if m:
            identifiers["decision_no"] = snap_decision_number(m.group(1))

    if not identifiers.get("decision_no") and active_case_id:
        if any(pronoun in q.lower() for pronoun in RELATIVE_PRONOUNS):
            identifiers["decision_no"] = active_case_id.replace("decision_", "")
            identifiers["_inferred_from_context"] = True

    m = re.search(r"(nkp[_\-][0-9]+(?:[_\-][0-9]+)?(?:[_\-]part[0-9]+)?\.pdf)", q, re.I)
    if m:
        identifiers["source"] = m.group(1)

    numbers = re.findall(r"\b([0-9]{3,5})\b", normalized)
    if len(numbers) >= 2:
        identifiers["multiple_decision_nos"] = [snap_decision_number(n) for n in numbers]

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
        "page_text": meta.get("page_text", meta.get("content", "")),
        "decision_no": meta.get("decision_no", ""),
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

def perform_hybrid_search(query, collection, model, bm25, chunk_metadata, top_k=5, alpha=0.15, current_case=None, identifiers=None):
    if identifiers is None:
        identifiers = extract_query_identifiers(query)

    total = len(chunk_metadata)
    if total == 0:
        return []

    intent = detect_query_intent(query)
    target_case_id = None
    target_decision_no = None

    lookups = get_lookup_indexes(chunk_metadata)
    candidate_set = None

    if identifiers.get("decision_no"):
        target_decision_no = snap_decision_number(identifiers["decision_no"])
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
    else:
        candidate_indices = None
        candidate_set = None

    if intent in ("CASE_SUMMARY", "CASE_LOOKUP", "CASE_ABOUT", "LEGAL_PRINCIPLE", "FACTUAL") and candidate_indices:
        header_idx = lookups["headers"].get(target_case_id)
        content_indices = [i for i in candidate_indices if i != header_idx]
        content_indices.sort(key=lambda i: chunk_metadata[i].get("page", 999))
        limit = max(top_k * 2, 10)
        content_indices = content_indices[:limit]
        result_indices = ([header_idx] if header_idx is not None else []) + content_indices
        results = []
        for idx in result_indices:
            meta = chunk_metadata[idx]
            r = _make_result(meta, 1.0, {"vector_score": 1.0, "bm25_score": 1.0, "lexical_score": 1.0})
            r["index"] = idx
            results.append(r)
        return results

    search_n = min(max(top_k * BM25_CANDIDATE_MULTIPLIER, 20), total if candidate_indices is None else len(candidate_indices))
    vector = encode_query(model, query)

    corpus_size = collection.count()
    safe_n = max(1, min(search_n * 3, VECTOR_CANDIDATE_CAP, corpus_size))
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
    if vec_res.get("ids"):
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

    query_tokens = char_ngram_tokenize(query)
    if hasattr(bm25, "search"):
        bm25_top, sparse_scores = bm25.search(query_tokens, top_k=search_n, allowed_indices=candidate_indices)
        union = set(vec_scores) | set(bm25_top)
        if target_case_id:
            header_idx = lookups["headers"].get(target_case_id)
            if header_idx is not None:
                union.add(header_idx)
        missing = [i for i in union if i not in sparse_scores]
        if missing:
            sparse_scores.update(bm25.score_docs(query_tokens, missing))
        bm25_for_doc = sparse_scores
    else:
        bm25_scores_raw = bm25.get_scores(query_tokens)
        bm25_top = _top_n_from_scores(bm25_scores_raw, search_n, candidate_indices)
        bm25_for_doc = {i: float(bm25_scores_raw[i]) for i in bm25_top}
        union = set(vec_scores) | set(bm25_top)
        if target_case_id:
            header_idx = lookups["headers"].get(target_case_id)
            if header_idx is not None:
                union.add(header_idx)
        for i in union:
            if i not in bm25_for_doc:
                bm25_for_doc[i] = float(bm25_scores_raw[i])

    max_bm25 = max(bm25_for_doc.values()) if bm25_for_doc else 1.0
    if max_bm25 <= 0:
        max_bm25 = 1.0

    results = []
    for i in union:
        meta = chunk_metadata[i]
        vector_score = vec_scores.get(i, 0.0)
        bm_score = max(0.0, float(bm25_for_doc.get(i, 0.0))) / max_bm25
        lexical = _lexical_score(query, meta.get("content", ""))
        fused = alpha * vector_score + (1 - alpha) * bm_score
        fused += 0.20 * lexical
        if _as_bool(meta.get("is_header", False)):
            fused += 2.0

        r = _make_result(meta, fused, {"vector_score": vector_score, "bm25_score": bm_score, "lexical_score": lexical})
        r["index"] = i
        results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)

    expanded = []
    seen_indices = set()
    for res in results[:top_k]:
        idx = res["index"]
        if idx not in seen_indices:
            expanded.append(res)
            seen_indices.add(idx)
        neighbour_score = res["score"] * 0.8
        neighbour_extra = {
            "vector_score": res["vector_score"] * 0.8,
            "bm25_score": res["bm25_score"] * 0.8,
            "lexical_score": res["lexical_score"] * 0.8,
        }
        for neighbour_idx in (idx - 1, idx + 1):
            if neighbour_idx < 0 or neighbour_idx >= total or neighbour_idx in seen_indices:
                continue
            neighbour_meta = chunk_metadata[neighbour_idx]
            if neighbour_meta.get("case_id") == res["case_id"]:
                nr = _make_result(neighbour_meta, neighbour_score, neighbour_extra)
                nr["index"] = neighbour_idx
                expanded.append(nr)
                seen_indices.add(neighbour_idx)

    expanded.sort(key=lambda x: x["score"], reverse=True)
    return expanded[:top_k]