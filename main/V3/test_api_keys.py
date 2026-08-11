"""
Quick verification that OMDb and YouTube Data API keys actually work.
Makes one cheap real call to each, not just a presence check.

Usage:
    python test_api_keys.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()


def test_omdb():
    # Accept either name in case of a naming mismatch between .env and this script.
    key = os.getenv("OMDB_API_KEY") or os.getenv("OMBD_API_KEY")
    if not key:
        print("OMDb: MISSING — no OMDB_API_KEY or OMBD_API_KEY found in .env")
        return

    response = requests.get("http://www.omdbapi.com/", params={"apikey": key, "t": "Friends"})
    data = response.json()
    if response.status_code == 200 and data.get("Response") == "True":
        print(f"OMDb: OK — fetched '{data.get('Title')}' ({data.get('Year')})")
    else:
        print(f"OMDb: FAILED — status={response.status_code} body={data}")


def test_youtube():
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        print("YouTube: MISSING — no YOUTUBE_API_KEY found in .env")
        return

    # Cheap call: videos.list costs 1 quota unit, vs search.list at 100.
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "snippet", "id": "dQw4w9WgXcQ", "key": key},
    )
    data = response.json()
    if response.status_code == 200 and data.get("items"):
        title = data["items"][0]["snippet"]["title"]
        print(f"YouTube: OK — fetched video title '{title}'")
    else:
        print(f"YouTube: FAILED — status={response.status_code} body={data}")


if __name__ == "__main__":
    test_omdb()
    test_youtube()
