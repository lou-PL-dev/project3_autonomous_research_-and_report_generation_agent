"""Node 5: generate — writes the recap draft, including the TMDB/cast-CSV
participant merge logic.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from openai import OpenAI

from config import OPENAI_CHAT_MODEL
from cost_tracker import record_usage
from season_index import load_cast_lookup

logger = logging.getLogger(__name__)


def node_generate(state: dict) -> dict:
    client = OpenAI()
    edition, season, episode, phase = state["edition"], state["season"], state["episode"], state["phase"]

    pods_phase_rule = ""
    if phase in ("Pods", "unknown"):
        pods_phase_rule = f"""

CRITICAL PODS-PHASE RULE: episode {episode}'s phase is "{phase}". Treat this conservatively
as Pods. Do not reveal which people end up together, engaged, or paired as couples, even if
sources describe final pairings. "participants" should still list the individuals who appear
in this episode's content, name/age/profession only, never who they're paired with.
"""

    retry_note = ""
    if state.get("spoiler_issues"):
        issues = "; ".join(state["spoiler_issues"])
        retry_note = f"""

PREVIOUS ATTEMPT FAILED THE SPOILER CHECK. Specific issues found: {issues}
Regenerate excluding these, using only the context provided below."""

    system_prompt = f"""You are a comic, dramatic soap-opera narrator writing a "previously on"
recap for the reality show Love Is Blind {edition}, Season {season}.

STRICT RULE: only use information about events up to and including episode {episode}.
Never mention or hint at anything past episode {episode}, even if it appears in the context.
{pods_phase_rule}{retry_note}

GROUNDING RULES, follow these exactly:
- Every name must be copied character-for-character as spelled in the context. If spelled
  differently across sources, use whichever spelling appears most often. Never guess.
- Do not state any specific claim unless it appears explicitly in the context.
- If the context is thin on a topic, keep that part general rather than inventing specifics.

Write like an excited friend texting another friend about the show, or a YouTube commenter
hyped about the drama, not a formal narrator. Conversational, casual, genuinely excited.
Occasional ALL CAPS and exclamation points where natural. No emojis. Avoid flowery vocabulary
(whirlwind, swirling, tangled web(s), rollercoaster, tapestry, saga, riveting, utterly, ablaze).

The context is organized into four labeled sections: PARTICIPANT BIOS, SEASON-WIDE DRAMA,
EPISODE {episode} SPECIFIC EVENTS, and AUDIENCE REACTION. Use each for its matching field.

Return ONLY valid JSON, no markdown fences, no preamble, with this exact shape:
{{
  "intro": "string, one short hype sentence",
  "main_drama": "string, everything that's happened THIS SEASON SO FAR up to {episode}, naming names",
  "highlights": {{
    "episode_number": {episode},
    "episode_title": "string or null",
    "moments": [{{"text": "string, 1-2 sentences with real detail", "drama_rank": 1}}]
  }},
  "audience_reaction": "string or null if nothing usable",
  "participants": [{{"name": "string", "age": "int or null", "profession": "string or null"}}],
  "sources": [{{"title": "string", "url": "string"}}],
  "conclusion": "string, one short closing sentence, teases what's next without spoiling"
}}

For "main_drama", use ONLY the SEASON-WIDE DRAMA section. Do NOT pull from EPISODE {episode}
SPECIFIC EVENTS, that section is reserved for "highlights" only. main_drama covers what
happened BEFORE episode {episode}, not episode {episode} itself.
For "highlights.moments", use ONLY the EPISODE {episode} SPECIFIC EVENTS section, 3-4 moments,
ranked by drama_rank (1 = most dramatic). A moment describes what happened, not how fans reacted.
For "audience_reaction", use the AUDIENCE REACTION section specifically.
For "participants", only include people who appear in the drama/episode content, use the
PARTICIPANT BIOS section only to fill in age/profession for those specific names, never to
introduce a name that doesn't otherwise appear. Name/age/profession only, no personality summary.

For "sources": this is not optional and one citation is almost never enough. After writing the
recap, go back through EACH of the four context sections (PARTICIPANT BIOS, SEASON-WIDE DRAMA,
EPISODE {episode} SPECIFIC EVENTS, AUDIENCE REACTION) one at a time and check whether anything
from that section ended up in your recap, if it did, that source belongs in the list. A recap
drawing on multiple sections should almost always cite multiple sources, one per section is a
reasonable floor, not a ceiling. Only exclude a source if you reviewed it and genuinely used
nothing from it at all.
"""

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL, temperature=0.4,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"CONTEXT:\n\n{state['context']}"}],
    )
    record_usage(OPENAI_CHAT_MODEL, response.usage)
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1)
    draft = json.loads(raw)
    # Ground-truth (season index) sources are real content the model may draw on,
    # but internal:// isn't a real, clickable URL, never show it to the user even
    # if the model cited it. Deterministic, not left to prompt compliance.
    draft["sources"] = [s for s in draft.get("sources", []) if not s.get("url", "").startswith("internal://")]

    # Structured, real YouTube-comment-based fan reaction overrides generate's own
    # simple string synthesis when available. The simple string stays as the
    # fallback for when no eligible (fully <= cutoff) YouTube video was found.
    if state.get("fan_reaction_analysis"):
        draft["audience_reaction"] = state["fan_reaction_analysis"]

    # Supplement participants with real TMDB names for THIS episode specifically
    # (not the whole 1-N range, confirmed that flattens into every contestant
    # who ever appeared, not who's actually still in the story). A match
    # ENRICHES the existing entry (fuller name, cast-CSV age/profession) rather
    # than just being skipped, otherwise a vague/wrong model-generated entry
    # (e.g. bare "Filip" with the wrong profession) survives untouched even
    # when TMDB had the correct disambiguated name sitting right there.
    # Age/profession only ever come from the cast CSV, never TMDB's role field,
    # "Self - Contestant" is not real profession data.
    def safe_name(name: str) -> str:
        # Structural spoiler guard: before the season has plausibly had a
        # wedding (Wedding/Reunion/After the Altar), cap a name at first name +
        # one surname. A married/compound surname is the concrete way a name
        # leaks a wedding outcome — confirmed case: a cast bio using a
        # post-show married name reached the draft by the model copying it
        # straight out of retrieved context, before the TMDB merge below ever
        # ran, so this has to guard every name-assignment site, not just the
        # merge's own override step.
        if phase in ("Wedding", "Reunion", "After the Altar"):
            return name
        tokens = name.split()
        return " ".join(tokens[:2]) if len(tokens) > 2 else name

    cast_lookup = load_cast_lookup(edition, season)
    current_ep_tmdb = state.get("tmdb_participants", {}).get(episode, [])
    participants = draft.setdefault("participants", [])
    for p in participants:
        p["name"] = safe_name(p["name"])

    def find_match_index(tmdb_name: str) -> Optional[int]:
        tmdb_lower = tmdb_name.lower()
        tmdb_tokens = tmdb_lower.split()
        for i, p in enumerate(participants):
            existing = p["name"].lower()
            if tmdb_lower in existing or existing in tmdb_lower:
                return i
            # First-NAME match only (index 0, not any shared token): catches
            # nickname/middle-name cases (e.g. "Kamil Uno" vs TMDB's "Kamil
            # Michał Osiak") without also matching on a shared SURNAME, which
            # incorrectly cross-attributed two different people who happen to
            # share a last name (confirmed case: TMDB's "Cameron Hamilton"
            # matching onto "Lauren Speed Hamilton" via the "Hamilton" token,
            # a married couple, overwriting her age/profession with his).
            existing_tokens = existing.split()
            if tmdb_tokens and existing_tokens and tmdb_tokens[0] == existing_tokens[0]:
                return i
        return None

    def find_cast_info(tmdb_name: str) -> dict:
        tmdb_lower = tmdb_name.lower()
        if tmdb_lower in cast_lookup:
            return cast_lookup[tmdb_lower]
        # Fallback: TMDB and the cast CSV sometimes use different naming
        # conventions for the same person (e.g. TMDB's "Julia Maria" vs the
        # CSV's "Julia Dumańska", first+middle vs first+last). Try a
        # first-token match, only if it resolves to exactly one candidate,
        # to avoid misattributing data between two different people who
        # happen to share a first name.
        tmdb_first = tmdb_lower.split()[0] if tmdb_lower.split() else ""
        candidates = [v for k, v in cast_lookup.items() if k.split() and k.split()[0] == tmdb_first]
        return candidates[0] if len(candidates) == 1 else {}

    for person in current_ep_tmdb:
        tmdb_name = safe_name(person["name"])
        is_host = "host" in person.get("role", "").lower()
        info = find_cast_info(tmdb_name)
        match_idx = find_match_index(tmdb_name)

        if match_idx is not None:
            existing = participants[match_idx]
            name_upgraded = len(tmdb_name.split()) > len(existing["name"].split())
            if name_upgraded:
                existing["name"] = tmdb_name
            existing["is_host"] = is_host
            # Once the identity is confirmed/disambiguated via TMDB, cast CSV
            # data is authoritative for THAT specific person, it overrides
            # rather than just fills gaps: the model's old age/profession may
            # have been attached to the wrong, ambiguous identity entirely
            # (e.g. "Filip" guessed as an Engineer, when the real Filip in
            # this episode, Filip Lenz, is a Flight Attendant per the CSV).
            # If the CSV has no entry for this exact person, leave whatever
            # the model already had rather than erasing it.
            if info.get("age"):
                existing["age"] = info["age"]
            if info.get("profession"):
                existing["profession"] = info["profession"]
        else:
            participants.append({
                "name": tmdb_name,
                "age": info.get("age"),
                "profession": info.get("profession"),
                "is_host": is_host,
            })

    # Final dedup pass: the model's own draft (before this merge ever ran) can
    # independently name the same person twice under different spellings
    # pulled from different context sections (e.g. ground-truth bios say
    # "Lauren Speed", other sources say "Lauren Speed Hamilton") — the merge
    # loop above only reconciles TMDB names against existing entries, it
    # doesn't catch two pre-existing draft entries that were already
    # duplicates of each other going in. Keep the longer (more complete) name
    # whenever one is a substring of the other.
    deduped: list = []
    for p in sorted(participants, key=lambda p: -len(p["name"])):
        p_lower = p["name"].lower()
        if any(p_lower in kept["name"].lower() for kept in deduped):
            continue
        deduped.append(p)
    draft["participants"] = deduped

    logger.info("generate attempt %d", state.get("attempts", 0) + 1)
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1}
