
import os
import requests
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError(
        "YOUTUBE_API_KEY not found.\n"
        "Make sure your .env file contains:\n\n"
        "YOUTUBE_API_KEY=your_key_here"
    )

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

SHOW = "Love Is Blind Poland"
SEASON = "Season 1"
EPISODE = 5

MAX_VIDEOS = 8
COMMENTS_PER_VIDEO = 20


# ============================================================
# YOUTUBE SEARCH
# ============================================================

def search_youtube(query, max_results=8):
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
        "key": API_KEY,
    }

    response = requests.get(SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    return response.json().get("items", [])


# ============================================================
# FETCH COMMENTS
# ============================================================

def get_comments(video_id, max_results=20):
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "order": "relevance",
        "textFormat": "plainText",
        "key": API_KEY,
    }

    response = requests.get(COMMENTS_URL, params=params, timeout=30)

    # Some videos have comments disabled.
    if response.status_code != 200:
        return []

    data = response.json()

    comments = []

    for item in data.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]

        comments.append(
            {
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
            }
        )

    return comments


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

def print_search_results(videos):
    print("\n" + "=" * 80)
    print("YOUTUBE SEARCH RESULTS")
    print("=" * 80)

    for i, video in enumerate(videos, start=1):
        snippet = video["snippet"]
        video_id = video["id"]["videoId"]

        print(f"\n{i}. {snippet['title']}")
        print(f"   Channel : {snippet['channelTitle']}")
        print(f"   Date    : {snippet['publishedAt']}")
        print(f"   Video ID: {video_id}")
        print(f"   URL     : https://www.youtube.com/watch?v={video_id}")
        print(f"   Description:")
        print(f"   {snippet.get('description', '')[:300]}")


# ============================================================
# DISPLAY COMMENTS
# ============================================================

def print_comments(video, comments):
    snippet = video["snippet"]
    video_id = video["id"]["videoId"]

    print("\n" + "-" * 80)
    print(f"COMMENTS: {snippet['title']}")
    print(f"https://www.youtube.com/watch?v={video_id}")
    print("-" * 80)

    if not comments:
        print("No comments available.")
        return

    for i, comment in enumerate(comments, start=1):
        text = comment["text"].replace("\n", " ")

        print(
            f"{i:02d}. "
            f"👍 {comment['likes']} | "
            f"{comment['published_at']}\n"
            f"    {text}"
        )


# ============================================================
# BASIC ANALYSIS
# ============================================================

def analyze_comments(all_comments):
    print("\n" + "=" * 80)
    print("BASIC COMMENT ANALYSIS")
    print("=" * 80)

    total_comments = sum(len(comments) for comments in all_comments.values())

    print(f"\nVideos analyzed : {len(all_comments)}")
    print(f"Comments fetched: {total_comments}")

    # Very simple episode-number signal.
    # This is deliberately NOT presented as a real spoiler classifier.
    later_episode_terms = [
        "episode 6",
        "episode 7",
        "episode 8",
        "episode 9",
        "episode 10",
        "episode 11",
        "ep 6",
        "ep 7",
        "ep 8",
        "ep 9",
        "ep 10",
        "ep 11",
        "e6",
        "e7",
        "e8",
        "e9",
        "e10",
        "e11",
    ]

    likely_later_episode = []

    for video_id, comments in all_comments.items():
        for comment in comments:
            text = comment["text"].lower()

            if any(term in text for term in later_episode_terms):
                likely_later_episode.append(comment)

    print(
        f"Comments explicitly mentioning later episodes: "
        f"{len(likely_later_episode)}"
    )

    if likely_later_episode:
        print("\nExamples:")
        for comment in likely_later_episode[:10]:
            print(f"\n⚠️ {comment['text']}")


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 80)
    print("YOUTUBE COMMENTS TEST")
    print("=" * 80)

    print(f"\nShow    : {SHOW}")
    print(f"Season  : {SEASON}")
    print(f"Episode : {EPISODE}")

    # --------------------------------------------------------
    # Test several query strategies
    # --------------------------------------------------------

    queries = [
        f"{SHOW} {SEASON} episode {EPISODE} reaction",
        f"{SHOW} {SEASON} episode {EPISODE} recap",
        f"{SHOW} E{EPISODE} reaction",
        f"{SHOW} episodes 1-{EPISODE} recap",
    ]

    all_videos = {}

    print("\nSearching YouTube...")

    for query in queries:

        print(f"\nQuery: {query}")

        videos = search_youtube(
            query=query,
            max_results=MAX_VIDEOS,
        )

        for video in videos:
            video_id = video["id"]["videoId"]
            all_videos[video_id] = video

    videos = list(all_videos.values())

    print(f"\nUnique videos found: {len(videos)}")

    print_search_results(videos)

    # --------------------------------------------------------
    # Fetch comments
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FETCHING COMMENTS")
    print("=" * 80)

    all_comments = {}

    for video in videos:

        video_id = video["id"]["videoId"]

        print(f"\nFetching comments for: {video['snippet']['title']}")

        comments = get_comments(
            video_id=video_id,
            max_results=COMMENTS_PER_VIDEO,
        )

        all_comments[video_id] = comments

        print(f"  → {len(comments)} comments fetched")

    # --------------------------------------------------------
    # Display comments
    # --------------------------------------------------------

    for video in videos:
        video_id = video["id"]["videoId"]

        print_comments(
            video,
            all_comments.get(video_id, []),
        )

    # --------------------------------------------------------
    # Basic analysis
    # --------------------------------------------------------

    analyze_comments(all_comments)

    # --------------------------------------------------------
    # Final research questions
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("WHAT TO EVALUATE")
    print("=" * 80)

    print(
        """
1. Are the search results actually about Episode 5?

2. Do episode-specific videos have enough comments?

3. Are comments specific enough to identify actual drama/events?

4. Do comments look like genuine fan reactions rather than generic:
   "OMG", "I love them", "this show is crazy", etc.?

5. Do comments mention specific participants?

6. Do comments contain useful disagreements or different opinions?

7. Do comments reveal later-episode information?

8. Are multi-episode videos more useful than episode-specific videos?

9. Are comments posted close enough to the episode release
   to represent the reaction at that point in the season?

10. Most importantly:
    Could these comments support a useful "Fan Reaction"
    section in the final product?
"""
    )


if __name__ == "__main__":
    main()
