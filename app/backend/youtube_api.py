import os
import re
import csv
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler

from bm25_engine import BM25Index, load_and_tokenize_csv, tokenize_and_expand_query


def minmax_normalize(values):
    """将数值列表线性缩放到 [0, 1]。若全部相同则返回 0.5 避免除零。"""
    arr = np.array(values, dtype=float).reshape(-1, 1)
    return MinMaxScaler().fit_transform(arr).flatten().tolist()


def log_minmax_normalize(values):
    """先做 log(1+x)，再缩放到 [0, 1]。适用于点赞数等非负稀疏值。"""
    log_arr = np.log1p(np.array(values, dtype=float)).reshape(-1, 1)
    return MinMaxScaler().fit_transform(log_arr).flatten().tolist()

#__file__ is the current file's path
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"

# It will load variables from the specified .env file into the OS environment variables (os.environ), for later code usage.
#only for this file
load_dotenv(_env_path)

api_key = os.environ.get("YOUTUBE_API_KEY")

BASE = "https://www.googleapis.com/youtube/v3"

# find frontend templates folder in app/frontend/templates and name it _frontend_templates
_frontend_templates = Path(__file__).resolve().parent.parent / "frontend" / "templates"

#use this file's path to create a Flask application instance and set the template folder to the frontend templates folder
app = Flask(__name__, template_folder=str(_frontend_templates))


def _data_dir():
    """Return app/data directory; all data paths (CSV, history, raw API) go through this."""
    return Path(__file__).resolve().parent.parent / "data"


# Colon in type hint is annotation only, not a default parameter
def search_videos(query: str, max_results: int):
    """
    Search videos by keyword on YouTube (API: search.list).
    in the official documentation of YouTube Data API v3, address: https://developers.google.com/youtube/v3/docs/search/list

    Args:
        query: Search keyword, e.g. "xiaomi projector 1s".
        max_results: Maximum number of videos to return .

    Returns:
        List of tuples: [(video_id, title), ...].
    """
    resp = requests.get(
        f"{BASE}/search",
        params={
            # only return snippet part of the video
            "part": "snippet",
            "type": "video",
            "maxResults": max_results,
            "q": query,
            "key": api_key,
        },
    )
    # Raises an HTTPError if the response status is not 2xx (successful)
    resp.raise_for_status()

    data = resp.json()

    # save raw API response to app/data/raw/api/
    api_raw_dir = _data_dir() / "raw" / "api"
    api_raw_dir.mkdir(parents=True, exist_ok=True)

    #make the query safe for file name
    safe_q = re.sub(r'[^\w\s\u4e00-\u9fff-]', "", query).strip()[:50]
    safe_q = re.sub(r"\s+", "_", safe_q)
    #get current timestamp
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    api_raw_path = api_raw_dir / f"{safe_q}_{ts}.json"
    api_raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    out = []
    for item in data.get("items", []):
        # each item has id.videoId (video ID) and snippet.title (title)
        vid = item.get("id", {}).get("videoId")
        title = item.get("snippet", {}).get("title", "")
        if vid:
            out.append((vid, title))
    return out

def get_comments(video_id: str, max_results: int):
    """
    Get top-level comments for a video (API: commentThreads.list).
    in the official documentation of YouTube Data API v3, address: https://developers.google.com/youtube/v3/docs/commentThreads/list

    Args:
        video_id: Video ID from search.list.
        max_results: Maximum number of comments to return.

    Returns:
        List of dicts: [{"author": "...", "text": "..."}, ...].
    """
    resp = requests.get(
        f"{BASE}/commentThreads",  
        params={
            "part": "snippet",      
            "videoId": video_id,   
            "maxResults": max_results,
            "textFormat": "plainText", 
            "key": api_key,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    # save commentThreads raw response to app/data/raw/comments/
    comments_raw_dir = _data_dir() / "raw" / "comments"
    comments_raw_dir.mkdir(parents=True, exist_ok=True)
    
    safe_vid = re.sub(r"[^\w-]", "", (video_id))[:20] or "unknown"
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    comments_raw_path = comments_raw_dir / f"{safe_vid}_{ts}.json"
    comments_raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    out = []
    for item in data.get("items", []):
        # Comment data is in snippet.topLevelComment.snippet
        top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        author = top.get("authorDisplayName", "")
        #textDisplay represents content with HTML format (like bold, emojis, etc.), and textOriginal represents the original user input content (pure text).
        text = top.get("textDisplay") if top.get("textDisplay") is not None else top.get("textOriginal", "")
        like_count = int(top.get("likeCount", 0))
        out.append({"author": author, "text": text, "like_count": like_count})
    return out
    


def _load_history():
    """return history list [{ keyword, csv_file, timestamp }, ...]"""
    history_path = _data_dir() / "history.json"
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        return data.get("items", [])
    #if the file is not found or the JSON is not valid, return an empty list
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _flatten_result(result):
    """Flatten API result to rows: each comment has video_id, video_url, video_title, author, text, like_count."""
    rows = []
    for v in result.get("videos", []):
        video_url = v.get("video_url") or f"https://www.youtube.com/watch?v={v.get('video_id', '')}"
        title = v.get("title") or ""
        for c in v.get("comments", []):
            rows.append({
                "video_id": v.get("video_id", ""),
                "video_url": video_url,
                "video_title": title,
                "author": c.get("author", ""),
                "text": c.get("text", ""),
                "like_count": c.get("like_count", 0),
            })
    return rows


def _run_search_and_save(keyword, max_videos=10, max_comments=100):
    """Execute YouTube search, write CSV, append history. On success return (result_dict, None), on failure (None, error_str)."""
    keyword = (keyword or "").strip()
    if not keyword:
        return None, "Please provide keyword parameter"
    try:
        videos = search_videos(keyword, max_results=max_videos)
    except requests.RequestException as e:
        return None, "YouTube search request failed: " + str(e)      
    result = {"query": keyword, "videos": []}
    
    for video_id, title in videos:
        try:
            comments = get_comments(video_id, max_results=max_comments)
        except requests.RequestException:
            comments = []
        result["videos"].append({
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title,
            "comments": comments,
        })
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_keyword = re.sub(r'[^\w\s\u4e00-\u9fff-]', "", keyword).strip() or "search"
    safe_keyword = re.sub(r"\s+", "_", safe_keyword)[:50]
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_filename = f"{safe_keyword}_{timestamp_str}.csv"
    csv_path = data_dir / csv_filename
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_title", "comment_text", "comment_likes"])
        for v in result["videos"]:
            title_safe = (v.get("title") or "").replace("\n", " ").replace("\r", "")
            for c in v["comments"]:
                writer.writerow([v["video_id"], title_safe, c.get("text", ""), c.get("like_count", 0)])
    result["csv_file"] = csv_filename
    history_path = data_dir / "history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        history = {"items": []}
    history["items"].insert(0, {
        "keyword": keyword,
        "csv_file": csv_filename,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, None


PER_PAGE = 20


@app.route("/")
def index():
    """Flask server-side render: search form, history list, result pagination."""
    keyword = request.args.get("keyword", "").strip()
    history_file = request.args.get("history_file", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    history = _load_history()

    rows = []
    query_label = None
    total = 0
    error = None
    current_file = None  # CSV for current result set, used by right-hand BM25 retrieval

    if keyword:
        result, err = _run_search_and_save(keyword)
        if err:
            error = err
        else:
            rows = _flatten_result(result)
            query_label = result.get("query", keyword)
            current_file = result.get("csv_file")
    elif history_file and ".." not in history_file and "/" not in history_file and "\\" not in history_file:
        data_dir = _data_dir()
        csv_path = data_dir / history_file
        #a iterator, find the given csv file in the history
        entry = next((x for x in history if x.get("csv_file") == history_file), None)
        if entry and csv_path.is_file():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("video_id", "")
                    like_count = int(row.get("comment_likes", row.get("like_count", 0)))
                    rows.append({
                        "video_id": vid,
                        "video_url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
                        "video_title": row.get("video_title", ""),
                        "author": "",
                        "text": row.get("comment_text", ""),
                        "like_count": like_count,
                    })
            query_label = entry.get("keyword", "")
            current_file = history_file

    total = len(rows)
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = min(max(1, page), total_pages)
    start = (page - 1) * PER_PAGE
    page_rows = rows[start : start + PER_PAGE]

    return render_template(
        "search.html",
        rows=page_rows,
        query=query_label,
        total=total,
        page=page,
        total_pages=total_pages,
        history=history,
        error=error,
        keyword_preserve=keyword or "",
        history_file_preserve=history_file or "",
        current_file=current_file or "",
    )


@app.route("/youtube", methods=["GET"])
def youtube_search():
    """API: GET /youtube?keyword=xxx returns JSON.""" 
    keyword = request.args.get("keyword", "").strip()
    if not keyword:
        return jsonify({"error": "Please provide keyword parameter"}), 400
    max_videos = min(int(request.args.get("max_videos", 10)), 50)
    max_comments = min(int(request.args.get("max_comments", 100)), 100)
    result, err = _run_search_and_save(keyword, max_videos=max_videos, max_comments=max_comments)
    if err:
        return jsonify({"error": err}), 502
    return jsonify(result)


@app.route("/history", methods=["GET"])
def get_history():
    """Return history list: keyword, csv_file, timestamp (API)."""
    items = _load_history()
    return jsonify({"items": items})


@app.route("/history/data", methods=["GET"])
def get_history_data():
    """Return comment data for the given csv_file (API)."""
    csv_file = request.args.get("file", "").strip()
    if ".." in csv_file or "/" in csv_file or "\\" in csv_file:
        return jsonify({"error": "Invalid file parameter"}), 400
    data_dir = _data_dir()
    items = _load_history()
    #ia iterator, find the given csv file in the history
    entry = next((x for x in items if x.get("csv_file") == csv_file), None)
    csv_path = data_dir / csv_file

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "")
            like_count = int(row.get("comment_likes", row.get("like_count", 0)))
            rows.append({
                "video_id": vid,
                "video_url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
                "video_title": row.get("video_title", ""),
                "author": "",
                "text": row.get("comment_text", ""),
                "like_count": like_count,
            })
    return jsonify({"query": entry.get("keyword", ""), "rows": rows})


@app.route("/retrieve", methods=["GET"])
def retrieve_bm25():
    """BM25 + likes fusion on current result set (specified CSV). GET /retrieve?q=xxx&file=yyy"""
    #get query and csv file from request parameters
    q = request.args.get("q").strip()
    csv_file = request.args.get("file").strip()

    lambda_param = 0.7
    data_dir = _data_dir()
    csv_path = data_dir / csv_file

    #load and tokenize the csv file
    rows, tokenized_docs = load_and_tokenize_csv(csv_path)

    like_counts = [int(r.get("comment_likes", r.get("like_count", 0))) for r in rows]
    max_likes = max(like_counts) or 1

    index = BM25Index(tokenized_docs)
    query_tokens = tokenize_and_expand_query(q)

    #max number is 200 and if the number of documents is less than 200, use the number of documents
    top_k = min(200, len(tokenized_docs))
    hits = index.search(query_tokens, top_k=top_k)

    #only get the scores of the hits(the first element of the tuple)
    bm25_scores = [s for _, s in hits]
    like_vals = [like_counts[doc_idx] for doc_idx, _ in hits]
    #normalize the scores
    bm25_norms = minmax_normalize(bm25_scores)
    likes_norms = log_minmax_normalize(like_vals)

    fused_list = []
    for i, (doc_idx, bm25_score) in enumerate(hits):
        lc = like_counts[doc_idx]
        fused_score = lambda_param * bm25_norms[i] + (1 - lambda_param) * likes_norms[i]
        fused_list.append((doc_idx, bm25_score, lc, fused_score))
    fused_list.sort(key=lambda x: -x[3])
    results = []
    
    for doc_idx, bm25_score, lc, fused_score in fused_list:
        row = rows[doc_idx]
        vid = row.get("video_id", "")
        text = row.get("comment_text", "") or ""
        snippet = (text[:120] + "…") if len(text) > 120 else text
        snippet = snippet.replace("\n", " ")
        results.append({
            "video_id": vid,
            "video_url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
            "video_title": row.get("video_title", ""),
            "text": text,
            "snippet": snippet,
            "like_count": lc,
            "bm25_score": round(bm25_score, 4),
            "fused_score": round(fused_score, 4),
            "score": round(fused_score, 4),
        })
    return jsonify({
        "query": q,
        "query_terms": query_tokens,
        "lambda": lambda_param,
        "results": results,
    })


if __name__ == "__main__":
    # Run on port 5000 locally; with debug=True code changes auto-reload
    app.run(host="127.0.0.1", port=5000, debug=True)