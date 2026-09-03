from src.sparse_index import SparseBM25
from src.text_processor import char_ngram_tokenize


def test_sparse_bm25_ranks_matching_doc_first():
    corpus = [
        char_ngram_tokenize("नागरिकता सम्बन्धी फैसला"),
        char_ngram_tokenize("अंशबण्डा र सम्पत्ति बाँडफाँड"),
        char_ngram_tokenize("उत्प्रेषण रिट र परमादेश"),
    ]
    index = SparseBM25.build(corpus)
    top_ids, scores = index.search(char_ngram_tokenize("उत्प्रेषण रिट"), top_k=2)
    assert top_ids[0] == 2
    assert scores[2] > scores.get(0, 0)


def test_sparse_bm25_respects_allowed_indices():
    corpus = [char_ngram_tokenize("निर्णय एक"), char_ngram_tokenize("निर्णय दुई")]
    index = SparseBM25.build(corpus)
    top_ids, _ = index.search(char_ngram_tokenize("निर्णय"), top_k=5, allowed_indices=[1])
    assert top_ids == [1]
