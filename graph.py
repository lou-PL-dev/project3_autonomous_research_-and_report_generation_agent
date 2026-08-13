"""RecapState, graph assembly, and the run_pipeline entry point."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from cost_tracker import check_budget_available, commit_run_to_ledger, reset_run
from generation import node_generate
from indexing import node_index
from logging_config import configure_logging
from metadata import node_fetch_show_metadata
from research import node_plan_and_search, node_rank_and_select
from season_index import node_load_season_index
from spoiler_check import node_spoiler_check, route_after_spoiler_check
from youtube import node_analyze_fan_reaction, node_fetch_youtube_comments
from retrieval import node_retrieve

logger = logging.getLogger(__name__)


def timed(name: str, fn, on_step=None):
    """Wrap a node function to log its wall-clock time, for locating the
    remaining bottlenecks after parallelizing the I/O-bound nodes. Also fires
    on_step(name) right before the node runs, this is the only hook point
    Flask's job runner needs to report live per-node progress to the UI.
    """
    @wraps(fn)
    def wrapper(state):
        if on_step:
            on_step(name)
        start = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - start
        logger.info("[timing] %s: %.2fs", name, elapsed)
        return result
    return wrapper


# Single source of truth for node order, shared with app.py so the job-status
# endpoint's progress fraction (step_index / len(NODE_ORDER)) can't drift out
# of sync with the actual graph. fetch_youtube_comments and index/retrieve run
# as parallel branches (see build_graph), so their relative firing order isn't
# guaranteed, this list is only used to compute a progress index, not to imply
# strict sequencing.
NODE_ORDER = [
    "fetch_show_metadata", "plan_and_search", "load_season_index", "rank_and_select",
    "fetch_youtube_comments", "index", "retrieve", "analyze_fan_reaction", "generate", "spoiler_check",
]


class RecapState(TypedDict):
    edition: str
    season: int
    episode: int
    phase: str
    episode_titles: dict[int, str]
    tmdb_participants: dict[int, list[dict]]
    raw_sources: list[dict]
    ground_truth_sources: list[dict]
    selected_sources: list[dict]
    youtube_comments: list[dict]
    fan_reaction_analysis: Optional[dict]
    chunks: list[dict]
    context: str
    draft: dict
    spoiler_issues: list[str]
    spoiler_passed: bool
    attempts: int


def build_graph(on_step=None):
    graph = StateGraph(RecapState)
    graph.add_node("fetch_show_metadata", timed("fetch_show_metadata", node_fetch_show_metadata, on_step))
    graph.add_node("plan_and_search", timed("plan_and_search", node_plan_and_search, on_step))
    graph.add_node("load_season_index", timed("load_season_index", node_load_season_index, on_step))
    graph.add_node("rank_and_select", timed("rank_and_select", node_rank_and_select, on_step))
    graph.add_node("fetch_youtube_comments", timed("fetch_youtube_comments", node_fetch_youtube_comments, on_step))
    graph.add_node("index", timed("index", node_index, on_step))
    graph.add_node("retrieve", timed("retrieve", node_retrieve, on_step))
    # defer=True: this node has two incoming branches of unequal length
    # (fetch_youtube_comments is 1 hop from rank_and_select, index->retrieve
    # is 2), so without deferring, LangGraph fires it as soon as the FIRST
    # branch's result lands rather than waiting for both, running it (and
    # everything downstream) twice. defer forces a real barrier: wait for
    # every upstream path that can still reach this node before running it.
    graph.add_node("analyze_fan_reaction", timed("analyze_fan_reaction", node_analyze_fan_reaction, on_step), defer=True)
    graph.add_node("generate", timed("generate", node_generate, on_step))
    graph.add_node("spoiler_check", timed("spoiler_check", node_spoiler_check, on_step))

    graph.set_entry_point("fetch_show_metadata")
    graph.add_edge("fetch_show_metadata", "plan_and_search")
    graph.add_edge("plan_and_search", "load_season_index")
    graph.add_edge("load_season_index", "rank_and_select")
    # fetch_youtube_comments and index->retrieve both depend only on
    # selected_sources, not on each other's output, so they fan out from
    # rank_and_select and run concurrently, joining at analyze_fan_reaction
    # (the first node that needs both youtube_comments and context). index is
    # the much longer chain (LLM tagging + embeddings + Pinecone), so this
    # hides the youtube fetch's wall-clock almost entirely behind it.
    graph.add_edge("rank_and_select", "fetch_youtube_comments")
    graph.add_edge("rank_and_select", "index")
    graph.add_edge("index", "retrieve")
    graph.add_edge("fetch_youtube_comments", "analyze_fan_reaction")
    graph.add_edge("retrieve", "analyze_fan_reaction")
    graph.add_edge("analyze_fan_reaction", "generate")
    graph.add_edge("generate", "spoiler_check")
    graph.add_conditional_edges("spoiler_check", route_after_spoiler_check, {"end": END, "retry": "generate"})

    return graph.compile()


def run_pipeline(edition: str, season: int, episode: int, on_step=None) -> dict:
    load_dotenv()
    configure_logging()
    check_budget_available()
    reset_run()
    app = build_graph(on_step)
    initial_state = {
        "edition": edition, "season": season, "episode": episode, "phase": "unknown",
        "episode_titles": {}, "tmdb_participants": {},
        "raw_sources": [], "ground_truth_sources": [], "selected_sources": [],
        "youtube_comments": [], "fan_reaction_analysis": None,
        "chunks": [], "context": "",
        "draft": {}, "spoiler_issues": [], "spoiler_passed": False, "attempts": 0,
    }
    try:
        final_state = app.invoke(initial_state)
        return final_state["draft"]
    finally:
        run_cost = commit_run_to_ledger()
        logger.info("[cost] this run: $%.4f", run_cost)
