"""
Minimal local API for the "Previously On: Love Is Blind" UI.
Wraps summary.py's fetch -> generate pipeline and reshapes the output
to match what the frontend expects.

Usage:
    python MVP/app.py
    (serves on http://localhost:5001)
"""

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from summary import build_context, fetch_sources, generate_recap

load_dotenv()

app = Flask(__name__)
CORS(app)  # the HTML file is opened as a local file, so allow cross-origin requests


def reshape_for_frontend(recap: dict) -> dict:
    """Convert summary.py's JSON shape into the shape the UI template expects."""
    moments = sorted(recap["highlights"]["moments"], key=lambda m: m["drama_rank"])
    ep_num = recap["highlights"]["episode_number"]
    ep_title = recap["highlights"]["episode_title"]
    episode_label = f"Episode {ep_num}" + (f": {ep_title}" if ep_title else "")

    participants = []
    for person in recap["participants"]:
        participants.append(
            {
                "name": person["name"],
                "age": person["age"] if person["age"] else "unknown",
                "occupation": person["profession"] if person["profession"] else "unknown",
            }
        )

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
        sources = fetch_sources(edition, season, episode)
        context = build_context(sources)
        recap = generate_recap(edition, season, episode, context)
        return jsonify(reshape_for_frontend(recap))
    except Exception as e:
        # Surface the error to the frontend instead of a silent 500 with no detail.
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
