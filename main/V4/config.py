"""Constants shared across the pipeline: model/index settings, chunking,
domain lists, and filesystem paths.
"""

import os

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
PINECONE_INDEX_NAME = "love-is-blind-recaps-v2"

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

# season_indexes/ lives one level up in main/, shared across all pipeline versions
SEASON_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "season_indexes")
