"""
Check what TMDB's per-episode credits endpoint actually returns for this show,
before deciding whether to build around it. Prints the real data shape, not a guess.

Usage:
    python main/V2/test_tmdb_episode.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_SHOW_ID = 312167  # Love Is Blind: Poland, confirmed from tonight's logs
SEASON = 1
EPISODE = 1


def main():
    key = os.getenv("TMDB_API_KEY")
    if not key:
        print("TMDB: MISSING — no TMDB_API_KEY found in .env")
        return

    url = f"https://api.themoviedb.org/3/tv/{TMDB_SHOW_ID}/season/{SEASON}/episode/{EPISODE}/credits"
    response = requests.get(url, params={"api_key": key})
    data = response.json()

    if response.status_code != 200:
        print(f"TMDB: FAILED — status={response.status_code} body={data}")
        return

    cast = data.get("cast", [])
    guest_stars = data.get("guest_stars", [])

    print(f"TMDB: OK — episode {EPISODE} credits fetched")
    print(f"\n'cast' field: {len(cast)} entries")
    for person in cast[:10]:
        print(f"    - {person.get('name')} as {person.get('character') or '(no character/role listed)'}")

    print(f"\n'guest_stars' field: {len(guest_stars)} entries")
    for person in guest_stars[:10]:
        print(f"    - {person.get('name')} as {person.get('character') or '(no character/role listed)'}")

    if not cast and not guest_stars:
        print("\nBoth fields are empty. TMDB has no per-episode cast data for this show/episode.")


if __name__ == "__main__":
    main()
