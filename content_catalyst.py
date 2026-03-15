"""
Content Catalyst — YouTube data fetch, preprocess, feature extraction, EDA.

Usage (example):
    python "c:/Users/navan/Downloads/capstone project/content_catalyst.py" --api_key YOUR_API_KEY --channel_id UC_x5XG1OV2P6uZZ5FSM9Ttw

Outputs:
 - outputs/summary.csv
 - outputs/summary.json
 - outputs/*.png (plots)
"""

import os
import argparse
import math
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from urllib.parse import urlparse

# Text processing
import re
import nltk

# Sentiment: try TextBlob then fallback to nltk VADER
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
VADER = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

# Create output dir
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# .env support: load environment variables from a local .env file if present.
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
def _load_dotenv_if_present(path: str = ENV_PATH):
    """Try to load a .env file. Prefer python-dotenv if installed, otherwise fall back to a simple parser.

    This will set variables into os.environ if they're not already present.
    """
    if not os.path.exists(path):
        return
    try:
        # prefer python-dotenv if available
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except Exception:
        pass

    # Manual, minimal .env parser (KEY=VALUE lines, ignores comments)
    try:
        with open(path, "r", encoding="utf8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # don't override existing env vars
                if k and os.environ.get(k) is None:
                    os.environ[k] = v
    except Exception:
        # don't raise — .env loading is convenience only
        return



def build_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def _extract_identifier_from_input(raw: str) -> str:
    """Normalize various channel identifier inputs into a likely channel identifier or username.

    Accepts full YouTube channel URLs, @handles, plain channel IDs (starting with UC), or usernames.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    # If it's a URL, extract the last path segment
    if s.startswith("http://") or s.startswith("https://"):
        try:
            p = urlparse(s)
            parts = [seg for seg in p.path.split("/") if seg]
            if parts:
                return parts[-1]
            return s
        except Exception:
            return s
    # If it looks like an @handle, strip @
    if s.startswith("@"):
        return s[1:]
    return s


def get_uploads_playlist_id(youtube, channel_identifier: str) -> str:
    """Resolve a channel identifier (id, username, handle, or URL) to the uploads playlist id.

    Tries in order:
    1. channels.list(id=...)
    2. channels.list(forUsername=...)
    3. search.list(q=..., type=channel) to find channelId
    Raises ValueError if the channel cannot be found.
    """
    ident = _extract_identifier_from_input(channel_identifier)
    # defensive: strip common prefixes that may appear in pasted values
    if ident.startswith("channel/"):
        ident = ident.split("/", 1)[1]
    if ident.startswith("user/"):
        ident = ident.split("/", 1)[1]

    # 1) Try interpreting as channel id (most direct)
    try:
        res = youtube.channels().list(part="contentDetails,snippet", id=ident).execute()
        items = res.get("items", [])
        if items:
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError as e:
        # capture details and continue to other strategies
        try:
            err_text = e.content.decode('utf8') if hasattr(e, 'content') else str(e)
        except Exception:
            err_text = str(e)
        print(f"Warning: channels.list(id=...) failed for '{ident}': {err_text}")

    # 2) Try as legacy username
    try:
        res = youtube.channels().list(part="contentDetails,snippet", forUsername=ident).execute()
        items = res.get("items", [])
        if items:
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError as e:
        try:
            err_text = e.content.decode('utf8') if hasattr(e, 'content') else str(e)
        except Exception:
            err_text = str(e)
        print(f"Warning: channels.list(forUsername=...) failed for '{ident}': {err_text}")

    # 3) Fallback: broad search for channels matching the identifier (try up to 5)
    try:
        sr = youtube.search().list(part="snippet", q=ident, type="channel", maxResults=5).execute()
        sitems = sr.get("items", [])
        if sitems:
            # prefer exact handle matches or the first reasonable result
            channel_id = None
            for it in sitems:
                snip = it.get("snippet", {})
                title = snip.get("title", "").lower()
                ch_id = snip.get("channelId")
                # exact title or id match
                if ident.lower() == title or ident == ch_id:
                    channel_id = ch_id
                    break
            if channel_id is None:
                channel_id = sitems[0]["snippet"].get("channelId")
            if channel_id:
                try:
                    res = youtube.channels().list(part="contentDetails,snippet", id=channel_id).execute()
                    items = res.get("items", [])
                    if items:
                        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
                except HttpError as e:
                    try:
                        err_text = e.content.decode('utf8') if hasattr(e, 'content') else str(e)
                    except Exception:
                        err_text = str(e)
                    print(f"Warning: channels.list(id={channel_id}) after search failed: {err_text}")
    except HttpError as e:
        try:
            err_text = e.content.decode('utf8') if hasattr(e, 'content') else str(e)
        except Exception:
            err_text = str(e)
        print(f"Warning: search.list(type=channel) failed for '{ident}': {err_text}")

    # If we reach here, provide an informative error including the identifier and a hint.
    raise ValueError(
        f"Channel not found for identifier: {channel_identifier}. "
        "Possible reasons: invalid identifier, API key lacks YouTube Data API access, quota exhausted, or the channel has no public uploads. "
        "Try passing a full channel URL, a @handle, or ensure YT_API_KEY is correct and has the YouTube Data API v3 enabled."
    )


def fetch_videos_from_channel(youtube, channel_id: str, max_results=500) -> List[Dict]:
    """
    Fetches video metadata (snippet) and statistics for a channel's uploads.
    Returns list of dicts with keys: video_id, title, tags, publishedAt, views, likes, comments.
    """
    playlist_id = get_uploads_playlist_id(youtube, channel_id)
    # quick verification: try to fetch the playlist itself to give a clearer error if it doesn't exist
    try:
        youtube.playlists().list(part="snippet", id=playlist_id, maxResults=1).execute()
    except HttpError as e:
        # If playlist lookup fails, raise with context
        raise ValueError(f"Uploads playlist '{playlist_id}' for channel '{channel_id}' could not be accessed: {e}") from e
    video_ids = []
    nextPageToken = None
    count = 0
    try:
        while True:
            pl_req = youtube.playlistItems().list(
                playlistId=playlist_id,
                part="contentDetails",
                maxResults=50,
                pageToken=nextPageToken
            )
            pl_res = pl_req.execute()
            for it in pl_res.get("items", []):
                video_ids.append(it["contentDetails"]["videoId"])
                count += 1
                if count >= max_results:
                    break
            nextPageToken = pl_res.get("nextPageToken")
            if not nextPageToken or count >= max_results:
                break
    except HttpError as e:
        # If playlist paging fails (404 or access error), fall back to searching videos by channelId
        print(f"Warning: could not page uploads playlist '{playlist_id}': {e}. Falling back to search by channelId.")
        # Use search.list with channelId to collect video IDs (paginated)
        try:
            nextToken = None
            while True:
                sr = youtube.search().list(part="id", channelId=channel_id, type="video", maxResults=50, pageToken=nextToken).execute()
                for it in sr.get("items", []):
                    vid = it.get("id", {}).get("videoId")
                    if vid:
                        video_ids.append(vid)
                        count += 1
                        if count >= max_results:
                            break
                nextToken = sr.get("nextPageToken")
                if not nextToken or count >= max_results:
                    break
        except Exception as e2:
            raise ValueError(f"Failed to retrieve videos for channel {channel_id} via playlist and search fallback: {e2}") from e2

    # Batch the video ids (50 per request)
    videos = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        vid_res = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch),
            maxResults=50
        ).execute()
        for v in vid_res.get("items", []):
            snippet = v.get("snippet", {})
            stats = v.get("statistics", {})
            videos.append({
                "video_id": v.get("id"),
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "tags": snippet.get("tags") or [],
                "publishedAt": snippet.get("publishedAt"),
                "categoryId": snippet.get("categoryId"),
                "viewCount": int(stats.get("viewCount", 0)) if stats.get("viewCount") is not None else np.nan,
                "likeCount": int(stats.get("likeCount", 0)) if stats.get("likeCount") is not None else np.nan,
                "dislikeCount": int(stats.get("dislikeCount", 0)) if stats.get("dislikeCount") is not None else np.nan,
                "commentCount": int(stats.get("commentCount", 0)) if stats.get("commentCount") is not None else np.nan,
                # CTR is only available via Analytics API; set as NaN
                "ctr": np.nan
            })
    return videos


def fetch_comments_for_video(youtube, video_id: str, max_results=200) -> List[str]:
    comments = []
    try:
        nextPage = None
        count = 0
        while True:
            res = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=nextPage
            ).execute()
            for it in res.get("items", []):
                top = it["snippet"]["topLevelComment"]["snippet"]
                comments.append(top.get("textDisplay", ""))
                count += 1
                if count >= max_results:
                    break
            nextPage = res.get("nextPageToken")
            if not nextPage or count >= max_results:
                break
    except HttpError:
        # comments disabled or not accessible
        pass
    return comments


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def preprocess_videos(videos: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(videos)
    # Ensure expected columns exist even if videos list is empty or incomplete
    expected_cols = [
        "video_id", "title", "description", "tags", "publishedAt", "categoryId",
        "viewCount", "likeCount", "dislikeCount", "commentCount", "ctr"
    ]
    for c in expected_cols:
        if c not in df.columns:
            # tags should be an empty list by default
            if c == "tags":
                df[c] = [[] for _ in range(len(df))]
            else:
                df[c] = np.nan

    # If still empty (no videos), return an empty DataFrame with the expected columns
    if df.shape[0] == 0:
        return df.reindex(columns=expected_cols)
    # Datetime
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["publish_date"] = df["publishedAt"].dt.date
    df["publish_hour"] = df["publishedAt"].dt.hour
    df["publish_weekday"] = df["publishedAt"].dt.day_name()
    df["is_weekend"] = df["publish_weekday"].isin(["Saturday", "Sunday"])

    # Numeric conversion and missing handling
    for col in ["viewCount", "likeCount", "commentCount", "dislikeCount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Engagement ratios (avoid divide by zero)
    df["likes_per_view"] = df.apply(lambda r: r["likeCount"]/r["viewCount"] if r["viewCount"] > 0 else 0.0, axis=1)
    df["comments_per_view"] = df.apply(lambda r: r["commentCount"]/r["viewCount"] if r["viewCount"] > 0 else 0.0, axis=1)
    df["likes_per_comment"] = df.apply(lambda r: (r["likeCount"]/(r["commentCount"] or 1)) if r["commentCount"] >= 0 else 0.0, axis=1)

    # Text cleaning
    df["clean_title"] = df["title"].fillna("").apply(clean_text)
    df["title_length_chars"] = df["title"].fillna("").apply(len)
    df["title_word_count"] = df["clean_title"].apply(lambda t: len(t.split()))
    df["tag_string"] = df["tags"].apply(lambda t: " ".join(t) if isinstance(t, list) else "")
    df["clean_tags"] = df["tag_string"].apply(clean_text)

    return df


def sentiment_score(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    if TEXTBLOB_AVAILABLE:
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            pass
    if VADER is not None:
        try:
            vs = VADER.polarity_scores(text)
            return vs.get("compound", 0.0)
        except Exception:
            pass
    # fallback simple heuristic
    positive_words = {"good", "great", "excellent", "love", "amazing", "best", "win"}
    negative_words = {"bad", "terrible", "worst", "hate", "awful", "fail"}
    tokens = set(text.split())
    score = (len(tokens & positive_words) - len(tokens & negative_words)) / max(1, len(tokens))
    return float(score)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sentiment on title and description
    df["title_sentiment"] = df["clean_title"].apply(sentiment_score)
    df["desc_sentiment"] = df["description"].fillna("").apply(lambda t: sentiment_score(clean_text(t)))

    # Keyword counts (simple counts of most common tokens in titles/tags)
    df["title_tokens"] = df["clean_title"].apply(lambda t: t.split())
    df["tag_tokens"] = df["clean_tags"].apply(lambda t: t.split())

    # Unique token counts
    df["title_unique_tokens"] = df["title_tokens"].apply(lambda toks: len(set(toks)))
    df["tag_unique_tokens"] = df["tag_tokens"].apply(lambda toks: len(set(toks)))

    return df


def top_keywords(df: pd.DataFrame, n=20) -> List[tuple]:
    all_tokens = []
    for toks in df["title_tokens"].tolist():
        all_tokens.extend(toks)
    for toks in df["tag_tokens"].tolist():
        all_tokens.extend(toks)
    cnt = Counter(all_tokens)
    return cnt.most_common(n)


def plot_correlation_heatmap(df: pd.DataFrame, filepath=os.path.join(OUT_DIR, "correlation_heatmap.png")):
    numeric_cols = ["viewCount", "likeCount", "commentCount", "title_length_chars", "title_word_count",
                    "likes_per_view", "comments_per_view", "title_sentiment"]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_views_vs_likes(df: pd.DataFrame, filepath=os.path.join(OUT_DIR, "views_vs_likes.png")):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df.sample(min(len(df), 1000)), x="viewCount", y="likeCount", alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Views (log)")
    plt.ylabel("Likes (log)")
    plt.title("Views vs Likes (log-log sample)")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_top_keywords_bar(top_kw, filepath=os.path.join(OUT_DIR, "top_keywords.png")):
    kws, counts = zip(*top_kw) if top_kw else ([], [])
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(counts), y=list(kws))
    plt.title("Top keywords across titles & tags")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_best_days_heatmap(df: pd.DataFrame, filepath=os.path.join(OUT_DIR, "weekday_hour_heatmap.png")):
    # pivot table average views by weekday and hour
    pivot = df.pivot_table(index="publish_weekday", columns="publish_hour", values="viewCount", aggfunc="mean").fillna(0)
    # Ensure weekdays ordering
    week_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(week_order).fillna(0)
    plt.figure(figsize=(14,6))
    sns.heatmap(np.log1p(pivot), cmap="YlGnBu")
    plt.title("Log-average views by weekday & hour")
    plt.ylabel("Weekday")
    plt.xlabel("Publish hour (0-23)")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_summary(df: pd.DataFrame, top_kw: List[tuple], out_prefix=os.path.join(OUT_DIR, "summary")):
    # summary metrics per video
    summary_cols = ["video_id", "title", "publishedAt", "viewCount", "likeCount", "commentCount",
                    "likes_per_view", "comments_per_view", "title_length_chars", "title_word_count",
                    "title_sentiment"]
    summary_df = df[summary_cols].copy()
    csv_path = out_prefix + ".csv"
    json_path = out_prefix + ".json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", force_ascii=False, lines=False)
    # metadata JSON with top keywords
    meta = {
        "top_keywords": [{"keyword": k, "count": c} for k,c in top_kw],
        "n_videos": len(df),
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(out_prefix + "_meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return csv_path, json_path, out_prefix + "_meta.json"


def generate_short_insights(df: pd.DataFrame, top_kw: List[tuple], top_n=5) -> str:
    insights = []
    # correlation glance
    corr = df[["viewCount", "likeCount", "commentCount", "title_word_count", "title_sentiment"]].corr()
    v_like = corr.loc["viewCount", "likeCount"]
    v_comment = corr.loc["viewCount", "commentCount"]
    insights.append(f"Views correlate with likes (r={v_like:.2f}) and comments (r={v_comment:.2f}).")
    # title sentiment effect
    s_corr = corr.loc["viewCount", "title_sentiment"]
    insights.append(f"Title sentiment correlation with views: r={s_corr:.2f}.")
    # title length
    t_corr = corr.loc["viewCount", "title_word_count"]
    insights.append(f"Title length (word count) correlation with views: r={t_corr:.2f}.")
    # top keywords
    topk = ", ".join([k for k,_ in top_kw[:top_n]])
    insights.append(f"Top keywords seen: {topk}.")
    # best days/hours
    pivot = df.pivot_table(index="publish_weekday", columns="publish_hour", values="viewCount", aggfunc="mean").fillna(0)
    avg_by_day = pivot.mean(axis=1).sort_values(ascending=False)
    best_day = avg_by_day.index[0]
    insights.append(f"Best average day to publish (by historical avg views): {best_day}.")
    return " ".join(insights)


def run_full_pipeline(api_key: str, channel_id: str, fetch_comments=False, max_videos=500, run_ml: bool = False):
    youtube = build_youtube_client(api_key)
    print("Fetching videos...")
    videos = fetch_videos_from_channel(youtube, channel_id, max_results=max_videos)
    if not videos:
        raise SystemExit(f"No videos found for channel '{channel_id}'. The channel may have no public uploads or the uploads playlist is inaccessible.")
    df = preprocess_videos(videos)
    df = extract_features(df)

    if fetch_comments:
        print("Fetching comments for sentiment aggregation (may be slow)...")
        all_comments_sentiments = []
        for idx, row in df.iterrows():
            comments = fetch_comments_for_video(youtube, row["video_id"], max_results=200)
            text = " ".join(comments)
            s = sentiment_score(clean_text(text))
            df.at[idx, "comments_sentiment"] = s
            all_comments_sentiments.append(s)
        if all_comments_sentiments:
            df["comments_sentiment"] = df["comments_sentiment"].fillna(0.0)
    else:
        df["comments_sentiment"] = 0.0

    print("Computing top keywords...")
    top_kw = top_keywords(df, n=30)

    print("Generating plots...")
    plot_correlation_heatmap(df)
    plot_views_vs_likes(df)
    plot_top_keywords_bar(top_kw[:20])
    plot_best_days_heatmap(df)

    print("Saving summary files...")
    csv_path, json_path, meta_path = save_summary(df, top_kw)

    print("Generating short insights...")
    summary_text = generate_short_insights(df, top_kw)
    # save a short text file
    with open(os.path.join(OUT_DIR, "insights.txt"), "w", encoding="utf8") as f:
        f.write(summary_text)

    ml_results = None
    if run_ml:
        try:
            # import local ml scaffolding (optional)
            from backend.ml import cluster_videos, train_views_predictor
        except Exception:
            try:
                # fallback if run from project root
                from ml import cluster_videos, train_views_predictor
            except Exception as e:
                print("ML modules not available; skipping ML steps:", e)
                cluster_videos = None
                train_views_predictor = None

        if cluster_videos is not None:
            try:
                clusters_out, _meta = cluster_videos(df, n_clusters=6)
                # save cluster metadata
                with open(os.path.join(OUT_DIR, "clusters.json"), "w", encoding="utf8") as cf:
                    json.dump(clusters_out, cf, ensure_ascii=False, indent=2)
                ml_results = ml_results or {}
                ml_results['clusters'] = os.path.join(OUT_DIR, "clusters.json")
            except Exception as e:
                print("Warning: cluster_videos failed:", e)

        if train_views_predictor is not None:
            try:
                metrics = train_views_predictor(df)
                # save metrics
                with open(os.path.join(OUT_DIR, "model_metrics.json"), "w", encoding="utf8") as mf:
                    json.dump(metrics, mf, ensure_ascii=False, indent=2)
                ml_results = ml_results or {}
                ml_results['model_metrics'] = os.path.join(OUT_DIR, "model_metrics.json")
            except Exception as e:
                print("Warning: train_views_predictor failed:", e)

    print("Done. Outputs saved to:", OUT_DIR)
    result = {
        "dataframe": df,
        "top_keywords": top_kw,
        "summary_csv": csv_path,
        "summary_json": json_path,
        "meta_json": meta_path,
        "insights": summary_text
    }
    if ml_results:
        result.update(ml_results)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Catalyst — YouTube EDA pipeline")
    parser.add_argument("--api_key", required=False, help="YouTube Data API v3 key (or set YT_API_KEY in env/.env)")
    parser.add_argument("--channel_id", required=False, help="YouTube channel ID (or set CHANNEL_ID in env/.env)")
    parser.add_argument("--max_videos", type=int, default=300, help="Max number of videos to fetch")
    parser.add_argument("--fetch_comments", action="store_true", help="Fetch comments for sentiment (slow)")
    args = parser.parse_args()

    # Load .env if present (will not override existing environment variables)
    _load_dotenv_if_present()

    api_key = args.api_key or os.environ.get("YT_API_KEY")
    channel_id = args.channel_id or os.environ.get("CHANNEL_ID")

    if not api_key:
        raise SystemExit("You must provide a YouTube API key via --api_key or set YT_API_KEY in the environment/.env file.")
    if not channel_id:
        raise SystemExit("You must provide a channel id via --channel_id or set CHANNEL_ID in the environment/.env file.")

    run_full_pipeline(api_key, channel_id, fetch_comments=args.fetch_comments, max_videos=args.max_videos)