"""
V2 API: wraps the ReAct planner + source-first RAG pipeline for the UI.
Same response shape as V1's app.py, so recap_ui.html works unchanged.

Usage:
    python app.py
    (serves on http://localhost:5004)
"""

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import csv
import os

from pipeline import run_pipeline

load_dotenv()

app = Flask(__name__)
CORS(app)
SEASON_INDEX_DIR = os.path.join(os.path.dirname(__file__), "season_indexes")


def load_phase_from_season_index(edition: str, season: int, episode: int) -> str | None:
    if edition.lower() != "poland" or season != 1:
        return None

    path = os.path.join(SEASON_INDEX_DIR, "poland_s1.csv")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["episode"]) == episode:
                    return row.get("phase") or None
    except (FileNotFoundError, KeyError, ValueError):
        return None

    return None


def format_audience_reaction(reaction) -> str:
    if not reaction:
        return "No fan reaction found in sources for this run."
    if isinstance(reaction, str):
        return reaction
    lines = [reaction.get("overall_reception", "")]
    if reaction.get("liked"):
        lines.append("\nWhat fans liked:")
        lines += [f"- {item}" for item in reaction["liked"]]
    if reaction.get("criticism"):
        lines.append("\nWhat is criticized:")
        lines += [f"- {item}" for item in reaction["criticism"]]
    if reaction.get("themes"):
        lines.append("\nMain themes:")
        lines += [f"- {item}" for item in reaction["themes"]]
    if reaction.get("sample_quotes"):
        lines.append("\nBest Quotes:")
        lines += [f"- \"{q.get('text', '')}\" ({q.get('context', '')})" for q in reaction["sample_quotes"]]
    return "\n".join(lines)


def reshape_for_frontend(recap: dict) -> dict:
    moments = sorted(recap["highlights"]["moments"], key=lambda m: m["drama_rank"])
    ep_num = recap["highlights"]["episode_number"]
    ep_title = recap["highlights"]["episode_title"]
    episode_label = f"Episode {ep_num}" + (f": {ep_title}" if ep_title else "")

    ordered = [p for p in recap["participants"] if not p.get("is_host")] + \
              [p for p in recap["participants"] if p.get("is_host")]
    participants = []
    for person in ordered:
        participants.append({
            "name": person["name"],
            "age": person["age"] if person["age"] else "unknown",
            "occupation": person["profession"] if person["profession"] else "unknown",
            "isHost": bool(person.get("is_host")),
        })

    return {
        "intro": recap["intro"],
        "mainDrama": recap["main_drama"],
        "episodeLabel": episode_label,
        "phase": recap.get("phase"),
        "highlights": [m["text"] for m in moments],
        "audienceReaction": format_audience_reaction(recap["audience_reaction"]),
        "participants": participants,
        "conclusion": recap["conclusion"],
        "sources": recap["sources"],
    }


@app.route("/api/recap")
def get_recap():
    edition = request.args.get("edition", "Poland")
    season = int(request.args.get("season", 1))
    episode = int(request.args.get("episode", 1))

    try:
        recap = run_pipeline(edition, season, episode)
        recap["phase"] = load_phase_from_season_index(edition, season, episode) or recap.get("phase")
        return jsonify(reshape_for_frontend(recap))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5004)
