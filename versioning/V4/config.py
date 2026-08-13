"""Constants shared across the pipeline: model/index settings, chunking,
domain lists, and filesystem paths.
"""

import os

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
PINECONE_INDEX_NAME = "love-is-blind-recaps-v2"

# Centralized so a model swap/cost experiment is a one-line change instead of
# a grep-and-replace across every node module.
OPENAI_CHAT_MODEL = "gpt-4o"        # generation, fan-reaction synthesis: quality-sensitive writing
OPENAI_MINI_MODEL = "gpt-4o-mini"   # research planning, chunk tagging, spoiler audit: cheaper/faster tasks

# USD per 1K tokens. Approximate, taken from published OpenAI pricing at the time
# this was written, not fetched live, re-check against OpenAI's pricing page if
# spend tracking (below) looks off.
MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}

# A run is refused before it starts if today's already-recorded spend has met
# or passed this. 0 (unset DAILY_BUDGET in .env) disables enforcement entirely.
DAILY_BUDGET = float(os.getenv("DAILY_BUDGET", "0") or 0)
SPEND_LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spend_ledger.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHARS_PER_SOURCE = 75000
MAX_SPOILER_RETRIES = 1
CATEGORY_BUDGETS = {"bios": 3, "highlights": 4, "drama": 4, "reaction": 3}
TAG_BATCH_SIZE = 30
UPSERT_BATCH_SIZE = 200

PHASES = ["Pods", "Honeymoon", "Moving In Together", "Wedding", "Reunion"]

GENERAL_DOMAINS = ["wikipedia.org", "themoviedb.org", "rottentomatoes.com", "imdb.com", "netflix.com"]
NOISE_DOMAINS = ["tiktok.com", "spotify.com"]

EDITION_ALIASES = {"poland": ["polska"]}
FRANCHISE_TERMS = ["love is blind", "casamento às cegas", "casamento as cegas"]  # franchise name across known localized titles

MAX_PLAUSIBLE_EPISODE = 20  # reality show seasons don't run this long; guards against
                             # matching a year (e.g. IMDb's "TV Episode 2026") as an episode number

# season_indexes/ lives at the repo root (shared with the root-level copy of
# this pipeline used for submission); this archived V4 copy reaches up two
# levels (versioning/V4 -> versioning -> repo root) to find it.
SEASON_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "season_indexes")
EPISODE_INDEX_DIR = os.path.join(SEASON_INDEX_DIR, "episodes")
CAST_INDEX_DIR = os.path.join(SEASON_INDEX_DIR, "cast")
