"""
V4 API: wraps the ReAct planner + source-first RAG pipeline (now modularized
across graph.py and its node modules) for the UI. Same response shape as
V1's app.py, so recap_ui.html works unchanged.

Usage:
    python app.py
    (serves on http://localhost:5004)
"""

from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import csv
import logging
import os
import threading
import time
import uuid
from typing import Optional

from config import EPISODE_INDEX_DIR
from cost_tracker import BudgetExceededError
from graph import NODE_ORDER, run_pipeline

load_dotenv()

app = Flask(__name__)
CORS(app)
logger = logging.getLogger(__name__)

# Friendly, in-voice labels for the UI's progress bar, keyed by graph.py's
# NODE_ORDER so the two can't drift apart silently.
STEP_LABELS = {
    "fetch_show_metadata": "Digging up the episode...",
    "plan_and_search": "Snooping around the internet...",
    "load_season_index": "Checking the plots...",
    "rank_and_select": "Sorting the tea...",
    "fetch_youtube_comments": "Reading the scandals...",
    "index": "Organizing the gossips...",
    "retrieve": "Pulling the juiciest bits...",
    "analyze_fan_reaction": "Feeling out the drama...",
    "generate": "Serving your recap...",
    "spoiler_check": "Slaying all spoilers...",
}

# In-memory job store for the async recap flow: /api/recap/start kicks a
# pipeline run off in a background thread (the pipeline itself stays fully
# synchronous, only the Flask request/response cycle stops blocking on it),
# /api/recap/status/<id> is polled by the UI for live per-node progress.
# Single-process, in-memory by design, this app has no multi-worker/multi-host
# deployment yet, if that changes this needs to move to something shared
# (Redis, a DB row) instead.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 1800


def _purge_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        stale = [job_id for job_id, job in _jobs.items() if job["created_at"] < cutoff]
        for job_id in stale:
            del _jobs[job_id]


def load_phase_from_season_index(edition: str, season: int, episode: int) -> Optional[str]:
    path = os.path.join(EPISODE_INDEX_DIR, f"{edition.lower()}_s{season}.csv")
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


@app.route("/api/library")
def get_library():
    """Scan season_indexes/episodes/ for which edition/season combos actually
    have a hand-verified index file, so the UI dropdowns never offer a combo
    that doesn't exist.
    """
    library: dict[str, list[int]] = {}
    for fname in os.listdir(EPISODE_INDEX_DIR):
        if not fname.endswith(".csv"):
            continue
        token, sep, season_part = fname[:-4].rpartition("_s")
        if not sep or not season_part.isdigit():
            continue
        library.setdefault(token, []).append(int(season_part))
    for seasons in library.values():
        seasons.sort()
    return jsonify(library)


@app.route("/api/recap")
def get_recap():
    """Synchronous, blocking recap generation. Kept as a simple direct path
    (curl/scripts/testing); recap_ui.html uses the async job endpoints below
    instead so the browser isn't left hanging on one open request for the
    full 30-90s pipeline run.
    """
    edition = request.args.get("edition", "Poland")
    season = int(request.args.get("season", 1))
    episode = int(request.args.get("episode", 1))

    try:
        recap = run_pipeline(edition, season, episode)
        recap["phase"] = load_phase_from_season_index(edition, season, episode) or recap.get("phase")
        return jsonify(reshape_for_frontend(recap))
    except BudgetExceededError as e:
        return jsonify({"error": str(e)}), 429
    except Exception as e:
        logger.exception("recap generation failed for %s season %s episode %s", edition, season, episode)
        return jsonify({"error": str(e)}), 500


def _run_job(job_id: str, edition: str, season: int, episode: int) -> None:
    def on_step(name: str) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["step"] = name
            job["step_label"] = STEP_LABELS.get(name, name)
            if name in NODE_ORDER:
                job["step_index"] = NODE_ORDER.index(name)

    try:
        recap = run_pipeline(edition, season, episode, on_step=on_step)
        recap["phase"] = load_phase_from_season_index(edition, season, episode) or recap.get("phase")
        payload = reshape_for_frontend(recap)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = payload
    except BudgetExceededError as e:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["error"] = str(e)
    except Exception as e:
        logger.exception("recap generation failed for %s season %s episode %s", edition, season, episode)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["error"] = str(e)


@app.route("/api/recap/start", methods=["POST"])
def start_recap():
    _purge_old_jobs()
    edition = request.args.get("edition", "Poland")
    season = int(request.args.get("season", 1))
    episode = int(request.args.get("episode", 1))

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "step": None,
            "step_label": "Starting up...",
            "step_index": 0,
            "total_steps": len(NODE_ORDER),
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
    threading.Thread(target=_run_job, args=(job_id, edition, season, episode), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/api/recap/status/<job_id>")
def recap_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "unknown or expired job"}), 404
        payload = {
            "status": job["status"],
            "step": job["step"],
            "stepLabel": job["step_label"],
            "stepIndex": job["step_index"],
            "totalSteps": job["total_steps"],
        }
        if job["status"] == "done":
            payload["result"] = job["result"]
        elif job["status"] == "error":
            payload["error"] = job["error"]
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, port=5004, threaded=True)
