"""RecapState, graph assembly, and the run_pipeline entry point."""

from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from generation import node_generate
from indexing import node_index
from metadata import node_fetch_show_metadata
from research import node_plan_and_search, node_rank_and_select
from season_index import node_load_season_index
from spoiler_check import node_spoiler_check, route_after_spoiler_check
from youtube import node_analyze_fan_reaction, node_fetch_youtube_comments
from retrieval import node_retrieve


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
    fan_reaction_analysis: dict | None
    chunks: list[dict]
    context: str
    draft: dict
    spoiler_issues: list[str]
    spoiler_passed: bool
    attempts: int


def build_graph():
    graph = StateGraph(RecapState)
    graph.add_node("fetch_show_metadata", node_fetch_show_metadata)
    graph.add_node("plan_and_search", node_plan_and_search)
    graph.add_node("load_season_index", node_load_season_index)
    graph.add_node("rank_and_select", node_rank_and_select)
    graph.add_node("fetch_youtube_comments", node_fetch_youtube_comments)
    graph.add_node("index", node_index)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("analyze_fan_reaction", node_analyze_fan_reaction)
    graph.add_node("generate", node_generate)
    graph.add_node("spoiler_check", node_spoiler_check)

    graph.set_entry_point("fetch_show_metadata")
    graph.add_edge("fetch_show_metadata", "plan_and_search")
    graph.add_edge("plan_and_search", "load_season_index")
    graph.add_edge("load_season_index", "rank_and_select")
    graph.add_edge("rank_and_select", "fetch_youtube_comments")
    graph.add_edge("fetch_youtube_comments", "index")
    graph.add_edge("index", "retrieve")
    graph.add_edge("retrieve", "analyze_fan_reaction")
    graph.add_edge("analyze_fan_reaction", "generate")
    graph.add_edge("generate", "spoiler_check")
    graph.add_conditional_edges("spoiler_check", route_after_spoiler_check, {"end": END, "retry": "generate"})

    return graph.compile()


def run_pipeline(edition: str, season: int, episode: int) -> dict:
    load_dotenv()
    app = build_graph()
    initial_state = {
        "edition": edition, "season": season, "episode": episode, "phase": "unknown",
        "episode_titles": {}, "tmdb_participants": {},
        "raw_sources": [], "ground_truth_sources": [], "selected_sources": [],
        "youtube_comments": [], "fan_reaction_analysis": None,
        "chunks": [], "context": "",
        "draft": {}, "spoiler_issues": [], "spoiler_passed": False, "attempts": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state["draft"]
