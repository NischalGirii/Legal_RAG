import math
from collections import Counter, defaultdict
import numpy as np


class SparseBM25:
    """Inverted-index BM25 (Okapi). Query time is posting-list size, not corpus size."""

    def __init__(self, n_docs, avgdl, doc_len, idf, postings, k1=1.5, b=0.75):
        self.n_docs = int(n_docs)
        self.avgdl = float(avgdl) or 1.0
        self.doc_len = np.asarray(doc_len, dtype=np.int32)
        self.idf = idf
        self.postings = postings
        self.k1 = float(k1)
        self.b = float(b)

    @classmethod
    def build(cls, tokenized_corpus, k1=1.5, b=0.75):
        n_docs = len(tokenized_corpus)
        doc_len = np.array([len(doc) for doc in tokenized_corpus], dtype=np.int32)
        avgdl = float(doc_len.mean()) if n_docs else 1.0
        df = defaultdict(int)
        tf_maps = []
        for tokens in tokenized_corpus:
            counts = Counter(tokens)
            tf_maps.append(counts)
            for term in counts:
                df[term] += 1

        idf = {}
        for term, freq in df.items():
            idf[term] = math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)

        postings = {}
        inverted = defaultdict(list)
        inverted_tf = defaultdict(list)
        for doc_id, counts in enumerate(tf_maps):
            for term, tf in counts.items():
                inverted[term].append(doc_id)
                inverted_tf[term].append(tf)
        for term in inverted:
            postings[term] = (
                np.asarray(inverted[term], dtype=np.int32),
                np.asarray(inverted_tf[term], dtype=np.float32),
            )
        return cls(n_docs, avgdl, doc_len, idf, postings, k1=k1, b=b)

    def _tf_for_doc(self, doc_arr, tf_arr, doc_id: int) -> float:
        pos = int(np.searchsorted(doc_arr, doc_id))
        if pos < doc_arr.size and int(doc_arr[pos]) == doc_id:
            return float(tf_arr[pos])
        return 0.0

    def _accumulate(self, doc_id: int, tf: float, idf: float, scores: dict) -> None:
        dl = float(self.doc_len[doc_id]) if doc_id < self.n_docs else self.avgdl
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
        scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1.0) / denom)

    def score_docs(self, query_tokens, doc_ids=None) -> dict[int, float]:
        term_counts = Counter(query_tokens)
        if doc_ids is not None:
            allowed_list = [int(i) for i in doc_ids]
            if len(allowed_list) <= 128:
                scores = {}
                for term, qf in term_counts.items():
                    posting = self.postings.get(term)
                    if posting is None:
                        continue
                    doc_arr, tf_arr = posting
                    idf = self.idf.get(term, 0.0) * qf
                    for doc_id in allowed_list:
                        tf = self._tf_for_doc(doc_arr, tf_arr, doc_id)
                        if tf:
                            self._accumulate(doc_id, tf, idf, scores)
                return scores

        allowed = None if doc_ids is None else set(int(i) for i in doc_ids)
        scores = {}
        for term, qf in term_counts.items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            idf = self.idf.get(term, 0.0) * qf
            doc_arr, tf_arr = posting
            for doc_id, tf in zip(doc_arr, tf_arr):
                doc_id = int(doc_id)
                if allowed is not None and doc_id not in allowed:
                    continue
                self._accumulate(doc_id, float(tf), idf, scores)
        return scores

    def search(self, query_tokens, top_k=20, allowed_indices=None) -> tuple[list[int], dict[int, float]]:
        if allowed_indices is not None and len(allowed_indices) == 0:
            return [], {}
        scores = self.score_docs(query_tokens, allowed_indices)
        if not scores:
            return [], {}
        items = list(scores.items())
        k = min(top_k, len(items))
        if k == len(items):
            items.sort(key=lambda x: x[1], reverse=True)
            ranked = items
        else:
            idx = np.argpartition([s for _, s in items], -k)[-k:]
            ranked = [items[i] for i in idx]
            ranked.sort(key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in ranked]
        return top_ids, scores


def retriever_from_pickle(data: dict):
    if data.get("sparse") is not None:
        return data["sparse"]
    return data.get("bm25")
