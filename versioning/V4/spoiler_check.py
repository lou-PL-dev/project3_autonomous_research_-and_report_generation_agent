"""Node 6: spoiler_check — LLM audit of the draft against the episode cutoff,
plus the retry-routing decision.
"""

import json
import logging

from openai import OpenAI

from config import MAX_SPOILER_RETRIES, OPENAI_MINI_MODEL
from cost_tracker import record_usage

logger = logging.getLogger(__name__)


def node_spoiler_check(state: dict) -> dict:
    client = OpenAI()
    episode = state["episode"]
    draft_text = json.dumps(state["draft"])

    prompt = f"""You are a spoiler-check auditor. The user has only watched up to and including
episode {episode}. Review this draft recap (JSON) and check whether it references anything
that would only be known from episode {episode + 1} onward.

CRITICAL RULE: content explicitly part of episode {episode} itself is NEVER a spoiler, no
matter how dramatic. A confession, a cheating reveal, a breakup, a confrontation are all fine
if they are the actual events of episode {episode}. Do not flag something just because it
sounds dramatic or implies future consequences in a general sense.

Only flag a claim that states or clearly implies a SPECIFIC fact confirmed to happen in
episode {episode + 1} or later, e.g. naming a wedding outcome before the Wedding phase, or
revealing pod pairings before the reveal.

CONCLUSION FIELD RULE: the "conclusion" field is deliberately written as a vague, generic
teaser ("can't wait to see what happens next", "as they prepare for what's ahead"). This is
intentional and REQUIRED by design, not a leak. Only flag the conclusion if it states a
SPECIFIC fact about a future episode (a name, an outcome, an event), not for containing
forward-looking phrasing in general.

Example of what is NOT a spoiler: "X admitted to cheating on Y with Z in episode {episode}."
Example of what is NOT a spoiler: "Can't wait to see how these relationships unfold next!"
Example of what IS a spoiler: "X and Y ultimately divorce" or "at the wedding, X says no."

Draft recap:
{draft_text}

Return ONLY JSON: {{"passed": true_or_false, "issues": ["specific issue 1", ...]}}
If passed is true, issues should be an empty array."""

    response = client.chat.completions.create(model=OPENAI_MINI_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}])
    record_usage(OPENAI_MINI_MODEL, response.usage)
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("spoiler-check response failed to parse as JSON, defaulting to passed=True")
        result = {"passed": True, "issues": []}

    logger.info("passed=%s issues=%s", result["passed"], result.get("issues"))
    return {"spoiler_passed": result["passed"], "spoiler_issues": result.get("issues", [])}


def route_after_spoiler_check(state: dict) -> str:
    if state["spoiler_passed"]:
        return "end"
    if state["attempts"] > MAX_SPOILER_RETRIES:
        logger.warning("max retries reached, returning draft as-is")
        return "end"
    return "retry"
