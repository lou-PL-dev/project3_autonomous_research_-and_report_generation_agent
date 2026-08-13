"""
V1 CLI: generate a spoiler-bounded recap using the RAG + LangGraph pipeline.

Usage:
    python main/V2/recap.py --episode 6
"""

import argparse

from pipeline import run_pipeline


def print_recap(edition: str, season: int, recap: dict) -> None:
    if recap.get("_spoiler_unverified"):
        print("⚠️  WARNING: this recap did not pass the automated spoiler check after retries.")
        print("   Read with caution, it may reference events beyond your requested episode.\n")

    print(f"=== Previously On: Love Is Blind {edition}, Season {season} ===\n")
    print(recap["intro"] + "\n")

    print("MAIN DRAMA SO FAR")
    print(recap["main_drama"] + "\n")

    ep_num = recap["highlights"]["episode_number"]
    ep_title = recap["highlights"]["episode_title"]
    header = f"HIGHLIGHTS OF THE LAST EPISODE (Episode {ep_num}"
    header += f": {ep_title})" if ep_title else ")"
    print(header)
    for moment in sorted(recap["highlights"]["moments"], key=lambda m: m["drama_rank"]):
        print(f"- {moment['text']}")
    print()

    print("AUDIENCE REACTION")
    reaction = recap["audience_reaction"]
    print(reaction if reaction else "No fan reaction found in sources for this run.")
    print()

    print("PARTICIPANTS")
    for person in recap["participants"]:
        age = person["age"] if person["age"] else "age unknown"
        profession = person["profession"] if person["profession"] else "profession unknown"
        print(f"{person['name']} ({age}, {profession})")
    print()

    print(recap["conclusion"])
    print()

    print("SOURCES")
    for source in recap["sources"]:
        print(f"- {source['title']}: {source['url']}")


def main():
    parser = argparse.ArgumentParser(description="Generate a Love Is Blind recap (V1, RAG + LangGraph).")
    parser.add_argument("--edition", type=str, default="Poland")
    parser.add_argument("--season", type=int, default=1)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    recap = run_pipeline(args.edition, args.season, args.episode)
    print_recap(args.edition, args.season, recap)


if __name__ == "__main__":
    main()