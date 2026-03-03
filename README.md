## YouTube-BM25-Ranker

### Dataset

- **Data source**: YouTube Data API v3 (keyword search for videos + video comments API).
- **Scale**: Up to 10 videos, up to 100 comments per video; aggregated into a single CSV file.
- **Fields**: CSV columns include `video_id`, `video_title`, `comment_text`, `comment_likes`; `comment_text` is the comment body, `comment_likes` is the like count used for usefulness fusion.

### Indexing and Representation

- **Document unit**: Each comment is one document; document text is the concatenation of `comment_text` and `video_title` to match both comment content and video title.
- **Tokenization and preprocessing**: Split on whitespace, lowercase, remove stopwords (NLTK English stopwords); **stemming** uses Porter's Algorithm (NLTK PorterStemmer) so document and query share the same stem space; **query expansion** expands stemmed query tokens with synonyms via NLTK WordNet (stems of synonyms are added) before BM25 retrieval.
- **Index**: BM25 index (rank_bm25) is built over **stemmed** tokenized documents; in-memory at runtime, data comes from the CSV above, no persisted inverted index file.

### Retrieval Model

- **BM25**: Standard BM25 with k1=1.5, b=0.75; top_k = min(200, total number of documents).
- **Usefulness signal**: Like counts are normalized with log(1+x) then min-max scaled; combined linearly with BM25 scores (min-max normalized over current hits); fusion formula: `FinalScore = λ·BM25_norm + (1-λ)·likes_norm`, λ=0.7.
- **Query processing**: Text normalization, stopword removal, Porter stemming, synonym expansion (WordNet), then BM25 retrieval on the expanded token list.

### Evaluation

- **No evaluation yet**: There is no labeled dataset or offline evaluation script; possible future work includes precision@k, recall, or manual sampling.

### Search Interface

- **Home page**: Keyword search (calls YouTube API to fetch videos and comments), history selection by CSV filename; paginated list of comments with like count, video link, and video title.
- **Right-hand BM25 retrieval**: Enter a query and select the current CSV; results are ranked by the fused score; displays snippet, like_count, bm25_score, fused_score; supports keyword highlighting and ranking explanation (λ·BM25 + (1-λ)·likes).

---

Author: nikk909 (yinghua253659@163.com)
