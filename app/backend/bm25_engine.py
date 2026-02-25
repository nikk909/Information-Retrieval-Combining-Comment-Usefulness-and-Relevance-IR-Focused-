"""
BM25 index and search: build index from tokenized_docs, return top-k (doc_idx, score) for query_tokens.
Uses rank_bm25.BM25Okapi for the BM25 scoring implementation.
"""
import csv
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

# download NLTK English version of stopwords
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
#list to set for faster lookup o(n) to o(1)
DEFAULT_STOP_WORDS = set(stopwords.words("english"))

# Create a Porter stemmer and save it as a module-level variable _PORTER_STEMMER.
_PORTER_STEMMER = PorterStemmer()


def tokenize_query(text: str) -> List[str]:
    """Same as document tokenization: split by space, lowercase, remove stopwords. For query use."""
    base_tokens = [
        #lower case and strip whitespace
        w.lower().strip()
        for w in (text or "").split()
        #remove stopwords and empty words
        if w.lower().strip() and w.lower().strip() not in DEFAULT_STOP_WORDS
    ]
    return [_PORTER_STEMMER.stem(w) for w in base_tokens]


def _get_raw_tokens(text: str) -> List[str]:
    """Same as tokenize_query but without stemming; used for WordNet lookup (needs word form)."""
    return [
        w.lower().strip()
        for w in (text or "").split()
        if w.lower().strip() and w.lower().strip() not in DEFAULT_STOP_WORDS
    ]


def expand_query_tokens_wordnet(raw_tokens: List[str]) -> List[str]:
    """Expand using NLTK WordNet: for each raw token get synonym lemmas, stem all; return deduped stemmed list."""
    out = set()
    for t in raw_tokens:
        out.add(_PORTER_STEMMER.stem(t))
        for syn in wordnet.synsets(t):
            for lemma in syn.lemmas():
                out.add(_PORTER_STEMMER.stem(lemma.name()))
    return list(out)


def load_and_tokenize_csv(csv_path: Path) -> Tuple[List[dict], List[List[str]]]:
    """
    Load documents from CSV, one comment per row; return (rows, tokenized_docs).
    CSV columns: video_id, video_title, comment_text.
    Tokenization: split by space, lowercase, remove stopwords.
    """
    rows = []
    tokenized_docs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            text = (row.get("comment_text", "") or "") + " " + (row.get("video_title", "") or "")
            tokens = [
                w.lower().strip()
                for w in text.split()
                if w.lower().strip() and w.lower().strip() not in DEFAULT_STOP_WORDS
            ]
            tokens = [_PORTER_STEMMER.stem(w) for w in tokens]
            tokenized_docs.append(tokens)
    return rows, tokenized_docs


def expand_query_tokens(query_tokens: List[str], synonym_map: dict = None) -> List[str]:
    """Return original tokens plus synonyms from synonym_map if given; deduplicated. When synonym_map is None, return tokens as-is."""
    if synonym_map is None:
        return list(query_tokens)
    out = list(query_tokens)
    for t in query_tokens:
        if t in synonym_map:
            for s in synonym_map[t]:
                if s not in out:
                    out.append(s)
    return out


def tokenize_and_expand_query(text: str, synonym_map: dict = None) -> List[str]:
    """Tokenize (with Porter stem), expand with WordNet synonyms; optionally add synonym_map expansion. For use in retrieve_bm25."""
    stemmed = tokenize_query(text)
    raw = _get_raw_tokens(text)
    wordnet_stems = expand_query_tokens_wordnet(raw)
    merged = list(dict.fromkeys(stemmed + [w for w in wordnet_stems if w not in stemmed]))
    return expand_query_tokens(merged, synonym_map)


class BM25Index:
    """
    BM25 index backed by rank_bm25.BM25Okapi.
    search() returns [(doc_idx, score), ...] sorted by score descending, at most top_k.
    """

    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = tokenized_docs
        self.N = len(tokenized_docs)
        self._bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)

    def search(self, query_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        """Search for query_tokens; return [(doc_idx, score), ...] sorted by score descending, at most top_k."""
        if not self.tokenized_docs or top_k <= 0:
            return []
        scores = self._bm25.get_scores(query_tokens)
        scored = [(i, float(s)) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


