"""
V4 CLI: generate a spoiler-bounded recap using the ReAct planner + source-first
RAG pipeline (now modularized across graph.py and its node modules).

Usage:
    python recap.py --episode 6
    python recap.py --edition UK --season 1 --episode 3
"""

import argparse

from cost_tracker import BudgetExceededError
from graph import run_pipeline


def print_recap(edition: str, season: int, recap: dict) -> None:
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
    if isinstance(reaction, dict):
        print(reaction.get("overall_reception", ""))
        if reaction.get("liked"):
            print("\nWhat fans liked:")
            for item in reaction["liked"]:
                print(f"- {item}")
        if reaction.get("criticism"):
            print("\nWhat is criticized:")
            for item in reaction["criticism"]:
                print(f"- {item}")
        if reaction.get("themes"):
            print("\nMain themes:")
            for item in reaction["themes"]:
                print(f"- {item}")
        if reaction.get("sample_quotes"):
            print("\nBest Quotes:")
            for q in reaction["sample_quotes"]:
                print(f"- \"{q.get('text', '')}\" ({q.get('context', '')})")
    else:
        print(reaction if reaction else "No fan reaction found in sources for this run.")
    print()

    print("PARTICIPANTS")
    contestants = [p for p in recap["participants"] if not p.get("is_host")]
    hosts = [p for p in recap["participants"] if p.get("is_host")]
    for person in contestants:
        age = person["age"] if person["age"] else "age unknown"
        profession = person["profession"] if person["profession"] else "profession unknown"
        print(f"{person['name']} ({age}, {profession})")
    for person in hosts:
        age = person["age"] if person["age"] else "age unknown"
        profession = person["profession"] if person["profession"] else "profession unknown"
        print(f"Host: {person['name']}, {age}, {profession}")
    print()

    print(recap["conclusion"])
    print()

    print("SOURCES")
    for source in recap["sources"]:
        print(f"- {source['title']}: {source['url']}")


def main():
    parser = argparse.ArgumentParser(description="Generate a Love Is Blind recap (V2: ReAct planner + source-first RAG).")
    parser.add_argument("--edition", type=str, default="Poland")
    parser.add_argument("--season", type=int, default=1)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    try:
        recap = run_pipeline(args.edition, args.season, args.episode)
    except BudgetExceededError as e:
        print(f"Run refused: {e}")
        return
    print_recap(args.edition, args.season, recap)


if __name__ == "__main__":
    main()