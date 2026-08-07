# Skills / Persistent Context for Agents

**What this is**: "Previously On: Love Is Blind" is an autonomous recap generator. A user gives an edition (e.g. Poland), season, and the last episode they watched. The system researches and generates a narrator-voiced recap covering the season so far, strictly bounded to that episode, no spoilers past it.

Phase structure: the show moves through phases in order: Pods → Honeymoon → Moving In Together → Wedding → Reunion. Reveal is not a separate phase, it happens during Pods. Spoiler safety is phase-aware, not just episode-number-aware, since two different episodes can fall in the same phase, and phase carries real meaning for what's safe to reveal (e.g. no couple pairings before the reveal).

What each recap section actually is:

intro: one short, hyped opening line for the whole recap. Sets tone, not content.
main_drama: the season-wide arc, everything that's happened across all episodes up to the cutoff, not a retelling of just the latest episode. This is where multiple couples/storylines get named and connected. Specific detail, not vague mood-setting.
highlights: distinct from main_drama, this is only about the requested episode itself. 3-4 ranked dramatic moments (most dramatic first), each with real detail (who, what, why it matters). Never fan reaction, that belongs in audience_reaction.
audience_reaction: how fans reacted, kept separate from plot narration. Can be null if sources don't cover it, that's a real, expected gap at this stage (proper fan-reaction data is a later-version feature via YouTube comments).
participants: name, age, profession only, no personality summaries. Only includes people who actually appear in the drama/episode content, a bio existing for someone isn't enough to include them. Special case: during the Pods phase (or when phase can't be confirmed), this field is replaced with a single "Wait for it!" placeholder instead of real names, since who paired with whom is itself a spoiler before the reveal.
sources: title + URL, but only sources actually used for a specific claim in this recap. A source that was fetched but didn't contribute anything must not be listed.
conclusion: one short closing line that teases what's next without spoiling it.

## Working agreement

1. **Explain before implementing.** Before writing or changing code, explain the
   fix and the reasoning behind it. Don't build first and explain after.
2. **Never assume scope. Always ask.** If a decision point isn't fully specified,
   ask for explicit input, even if an answer seems implied or obvious. Don't
   invent requirements to fill a gap.
3. **Report what actually changed.** After implementing, give a clear, specific
   account of the change, not a vague "fixed it." Say exactly what was modified
   and why.

## End goals to preserve, regardless of implementation

- **Spoiler safety is non-negotiable.** The user must never see content from
  past their stated episode cutoff. Every other goal is secondary to this one.
- **Grounding.** Never invent names, ages, professions, or plot claims that
  aren't explicitly present in retrieved source content. When evidence is thin
  or ambiguous, leave it out rather than guess.
- **Tone.** Conversational, hyped, "excited friend texting about the show" or
  YouTube-commenter voice. Not a formal or overwrought narrator. Avoid flowery
  vocabulary.
- **Structured, complete output.** intro, main drama so far (season-wide, not
  just the latest episode), highlights of the requested episode specifically,
  audience reaction, participants (name/age/profession), sources, conclusion.
- **Season-wide coverage.** The recap should reflect everything that's happened
  up to the user's cutoff, not just the most recent episode.
