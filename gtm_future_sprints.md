# Future GTM Sprints

These are the next sprints after this MVP or MLP (Minimum Loveable Product), aimed at go-to-market — not a
feature backlog. The MLP proves the core product (autonomous, spoiler-bounded, on-tone recap generation); these sprints are about
getting it in front of real users and finding out if they'll come back for
it unprompted.

**Overall motion: community-led GTM.** Primary channel is organic
participation in existing fan communities (Reddit — r/LoveIsBlindOnNetflix
and similar), not cold self-promotion or paid acquisition. Secondary
channel turns potential competitors (recap YouTubers/TikTokers) into
distribution partners instead of competing with them head-on for the same
audience.

---

## Sprint 1 — Community seeding on Reddit

- **Goal**: Validate that real fans outside the builder's own circle will
  use the recap generator unprompted and come back for it, and generate
  the first organic word-of-mouth loop.
- **Target user/buyer**: Active members of Love Is Blind fan subreddits
  (r/LoveIsBlindOnNetflix and edition-specific threads) — the same
  audience segment identified in the original plan (skews female, 35–64,
  watching for drama/emotional engagement).
- **Channel/motion**: Organic participation, not advertising. Show up as
  a real community member first (comments, episode-discussion threads),
  share the tool when it's genuinely relevant to a discussion (e.g. a
  "catching up, no spoilers past ep 6" thread), not as a drive-by promo
  post.
- **Key deliverable**: A public, shareable version of the tool (hosted
  web UI, not "clone this repo") with a link that can survive being
  dropped into a Reddit comment — fast enough and self-explanatory enough
  to work with zero onboarding.
- **Success metric**: Return usage — a tracked cohort of first-time users
  from Reddit who generate a second recap in a later week without being
  re-prompted. Secondary metric: organic mentions/shares of the tool by
  users who weren't directly invited.

## Sprint 2 — Creator partnership as a drafting tool

- **Goal**: Convert existing recap YouTubers/TikTokers from potential
  competitors into a distribution channel, by pitching the tool as a
  research/drafting aid for their own content rather than a replacement
  for it.
- **Target user/buyer**: Small-to-mid Love Is Blind recap creators on
  YouTube/TikTok — people who currently spend hours per episode manually
  compiling plot beats, cast facts, and fan reaction to script their
  videos.
- **Channel/motion**: Direct outreach to a shortlist of creators, offering
  early/free access in exchange for feedback and (if it's useful to them)
  an organic mention or credit in their content. This is partnership
  motion, not a paid sponsorship at this stage.
- **Key deliverable**: A "creator mode" output — the same underlying
  pipeline, but the report format optimized for scripting use (e.g. beat
  list with timestamps/sources instead of narrator prose) — and a short
  outreach kit (one-pager + sample output) to pitch with.
- **Success metric**: Number of creators who use it for ≥2 consecutive
  episodes of their own content without further prompting, and any
  resulting inbound traffic/signups attributable to a creator mention
  (tracked via a unique link per creator).

## Sprint 3 — Freemium launch

- **Goal**: Introduce the monetization layer and validate that a
  meaningful share of the user base from Sprints 1–2 will convert to a
  paid tier, without degrading the free core product that drove adoption.
- **Target user/buyer**: The most engaged users from Sprints 1–2 —
  repeat users and anyone arriving via a creator partner.
- **Channel/motion**: In-product upsell only (no separate paid
  acquisition channel yet) — free core product stays free, paid features
  are additive, not a paywall on the thing that got people in the door.
- **Key deliverable**: Paid tier shipping three concrete features:
  - **Share-with-a-friend program**: send a generated recap directly to a
    friend (who's watching the same show, possibly at a different episode)
    instead of only viewing it yourself — the natural distribution loop for
    a show people watch and text each other about in real time.
  - **Email this to yourself**: scheduled/on-demand delivery of a recap to
    your own inbox, so it's waiting for you the morning after an episode
    airs, no need to re-request it.
  - **Vote for who's most likely to get married**: a lightweight prediction
    game per season, one vote per user per couple, results revealed at the
    Wedding phase — pure engagement/retention feature, no bearing on the
    recap content itself.
  Free tier stays exactly what the MVP already does: on-demand recap
  generation.
- **Success metric**: Free-to-paid conversion rate on the cohort of users
  who've generated ≥2 free recaps, plus paid-tier retention (still
  subscribed/active after one full season).

---

## Product roadmap: true "After the Show" status feature

Not a GTM sprint — a genuine product feature deliberately scoped out of
the current MVP, worth documenting here because it's a different kind of
build than anything shipped so far and shapes what a later sprint (e.g.
around Sprint 3's paid tier) could offer.

**What's built today**: the recap pipeline supports an `"After the Altar"`
phase — later, still episode-numbered installments (e.g. "After the Altar:
...") that air after a season's Reunion special. These are treated like
any other phase: bound to the user's episode cutoff, fully spoiler-checked,
folded into the normal `main_drama`/`highlights` recap shape. (Renamed from
an earlier, ambiguous `"After the show"` label to free that name up for
the feature below — see `config.py`.)

**What's not built**: a true "as of today" status feature — "where are the
couples from this season now?" — answerable at any time, independent of
which episode the user has watched. This is a meaningfully different
feature, not an extension of the existing phase logic:

- **No episode cutoff, no spoiler check the way every other phase works.**
  The entire current pipeline is built around "don't reveal past episode
  N" — this feature is the opposite, it's explicitly about revealing
  present-day status regardless of episode progress.
- **Different output shape.** No `main_drama`, no `highlights`, no
  episode-anchored fields at all — instead something like: per-couple
  current status (together / broken up / other), with sourced, dated
  claims (a status is only as good as how recently it was reported).
  Same narrator voice, entirely different schema.
- **Different sourcing need.** Requires current, dated sources (recent
  interviews, social media, tabloid coverage), not the show's own episode
  content — closer to an ongoing news-monitoring problem than a one-time
  RAG index built from a fixed set of episodes.
- **Different trigger model.** Doesn't make sense as "trigger once per
  episode watched" — more naturally a standing query a user can re-run any
  time ("what's the latest on X and Y?"), which also means it ages: an
  answer from today may be wrong in a month, unlike the rest of the recap
  which is permanently correct once generated.

Scoping this as its own future sprint rather than folding it into the
current phase logic, given how much of the existing architecture (cutoff
filtering, spoiler-check, phase-based retrieval) simply doesn't apply to
it.

## Product roadmap: Reddit API as a fan-reaction source

Not built in the MVP for a documented reason (see
[`versioning/MVP/planning.md`](versioning/MVP/planning.md) §2): official
Reddit API access requires manual approval (2–4 week wait), and non-commercial
self-serve registration was closed as of 2025 — too fragile for a 5-day
build, so YouTube comments served as the fan-reaction source instead.

Worth revisiting once Sprint 1 is underway, for two reasons:

- **Better raw material than YouTube comments.** Subreddit discussion
  threads (r/LoveIsBlindOnNetflix and edition-specific ones) tend to be
  more detailed and less repetitive than YouTube comment sections —
  closer to real discourse than drive-by reactions, which should improve
  the quality of the `audience_reaction` section specifically.
- **Sprint 1 removes the biggest blocker to it.** Once there's a real,
  organic presence in these communities (the whole point of Sprint 1),
  applying for API access as an established participant rather than a
  cold, unknown request is both a more honest use of the access and
  plausibly a faster approval path.

Scope if picked up: an additional source alongside YouTube in the existing
`fetch_youtube_comments`/`analyze_fan_reaction` step (same range-gating
requirement applies — nothing from a subreddit thread discussing an
episode beyond the user's cutoff), not a replacement for it.

---

## Business model

**Freemium.**
- **Free**: on-demand recap generation — the core product, unchanged.
- **Paid**: scheduled/email delivery, social sharing, interactive features
  (tagging friends, couple predictions/voting).

## Known risk: IP / cease-and-desist exposure

Netflix/Love Is Blind is a trademark- and IP-sensitive franchise. A
cease-and-desist is a real possibility as usage grows beyond a personal
project.

**Position**: treat a C&D as a validation signal rather than a pure
downside — Amazon Prime Video's own X-Ray text summaries and AI-narrated
video recaps are precedent that this content category (spoiler-bounded,
personalized episode recaps) is viable and defensible as a product
category, even for IP the recapper doesn't own. A C&D means the product
got big enough to notice.

**Exit paths this opens, not forecloses**: an acquisition pitch to
Netflix/a Netflix-adjacent recap/companion-app business, or a hiring
signal (the builder demonstrated exactly the kind of fan-engagement
product a streamer's product team would want). Both are upside scenarios
contingent on Sprints 1–2 actually proving organic demand first — they are
not the plan if usage never gets there.
