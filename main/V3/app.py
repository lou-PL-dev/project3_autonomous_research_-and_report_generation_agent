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

from pipeline import run_pipeline

load_dotenv()

app = Flask(__name__)
CORS(app)


def reshape_for_frontend(recap: dict) -> dict:
    moments = sorted(recap["highlights"]["moments"], key=lambda m: m["drama_rank"])
    ep_num = recap["highlights"]["episode_number"]
    ep_title = recap["highlights"]["episode_title"]
    episode_label = f"Episode {ep_num}" + (f": {ep_title}" if ep_title else "")

    participants = []
    for person in recap["participants"]:
        participants.append({
            "name": person["name"],
            "age": person["age"] if person["age"] else "unknown",
            "occupation": person["profession"] if person["profession"] else "unknown",
        })

    return {
        "intro": recap["intro"],
        "mainDrama": recap["main_drama"],
        "episodeLabel": episode_label,
        "highlights": [m["text"] for m in moments],
        "audienceReaction": recap["audience_reaction"] or "No fan reaction found in sources for this run.",
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
        return jsonify(reshape_for_frontend(recap))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5004)
