"""Token-usage cost tracking and daily-budget enforcement across every OpenAI
call in the pipeline (chat completions, embeddings, and the LangChain-driven
ReAct agent). A run's cost accumulates in memory (thread-safe, since several
nodes fire parallel calls via ThreadPoolExecutor) and is committed to a
per-day JSON ledger once the run finishes.
"""

import json
import logging
import os
import threading
from datetime import date

from config import DAILY_BUDGET, MODEL_PRICING, SPEND_LEDGER_PATH

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_run_total = 0.0


class BudgetExceededError(RuntimeError):
    pass


def _cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        logger.warning("no pricing entry for model '%s', cost tracking will undercount", model)
        return 0.0
    return (prompt_tokens / 1000) * pricing["input"] + (completion_tokens / 1000) * pricing["output"]


def record(model: str, prompt_tokens: int, completion_tokens: int = 0) -> None:
    global _run_total
    cost = _cost_for(model, prompt_tokens, completion_tokens)
    with _lock:
        _run_total += cost


def record_usage(model: str, usage) -> None:
    """Convenience wrapper for an OpenAI SDK `usage` object, chat or embeddings."""
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    record(model, prompt_tokens, completion_tokens)


def record_langchain_usage(model: str, messages: list) -> None:
    """Best-effort extraction from LangChain AIMessage.usage_metadata, used for
    the ReAct planner agent, which doesn't expose one plain `usage` object the
    way a direct client.chat.completions.create() call does. Silently
    undercounts (with a debug log) rather than raising, since this is a
    convenience wrapper around an internal LangChain attribute that could
    change shape between versions.
    """
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if not usage:
            continue
        try:
            record(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        except AttributeError:
            logger.debug("could not read usage_metadata off message, skipping cost tracking for it")


def reset_run() -> None:
    global _run_total
    with _lock:
        _run_total = 0.0


def get_run_total() -> float:
    with _lock:
        return _run_total


def _load_ledger() -> dict:
    today = str(date.today())
    if not os.path.exists(SPEND_LEDGER_PATH):
        return {"date": today, "total": 0.0}
    try:
        with open(SPEND_LEDGER_PATH, encoding="utf-8") as f:
            ledger = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"date": today, "total": 0.0}
    if ledger.get("date") != today:
        return {"date": today, "total": 0.0}
    return ledger


def _save_ledger(ledger: dict) -> None:
    try:
        with open(SPEND_LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(ledger, f)
    except OSError as e:
        logger.warning("failed to persist spend ledger: %s", e)


def check_budget_available() -> None:
    """Call BEFORE starting a pipeline run."""
    if DAILY_BUDGET <= 0:
        return
    ledger = _load_ledger()
    if ledger["total"] >= DAILY_BUDGET:
        raise BudgetExceededError(
            f"Daily budget of ${DAILY_BUDGET:.2f} already spent (${ledger['total']:.2f} so far today), "
            "refusing to start a new run."
        )


def commit_run_to_ledger() -> float:
    """Call AFTER a pipeline run finishes (success or failure) to persist this
    run's actual spend into today's running total. Returns this run's cost.
    """
    run_cost = get_run_total()
    ledger = _load_ledger()
    ledger["total"] += run_cost
    _save_ledger(ledger)
    if DAILY_BUDGET > 0 and ledger["total"] >= DAILY_BUDGET:
        logger.warning("daily budget reached: $%.2f/$%.2f spent today", ledger["total"], DAILY_BUDGET)
    return run_cost
