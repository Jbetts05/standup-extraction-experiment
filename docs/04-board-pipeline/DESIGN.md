# POC design: stand-up → task board, with board context

<p align="center">
  <img src="board-loop.svg" alt="Stand-up 1: the board is empty, so Step 1 extracts three new jobs, Step 2 finds nothing on the board, and Step 3 creates three cards. Stand-up 2: for each thing said, Step 1 extracts who, what and what was said; Step 2 pulls that person's open cards and any cards on the same equipment; Step 3 updates the matching card, creates a card for a new job with no match, and ignores an update with no match." width="100%">
</p>

| | |
|---|---|
| **Status** | **Built as a local POC** (results just below). v0, the whole-board version, is kept alongside for comparison. |
| **Approach** | A fixed **pipeline**, not an agent. The steps always run in the same order, and each can be tested on its own. |
| **Runs** | Locally only. Nothing is deployed. The only cloud calls are to the existing Foundry model deployments. |
| **Data** | Synthetic stand-ups we generate ourselves. The existing `data/series` set is reused as-is. |
| **Models** | GPT-5.6 Luna for Step 1 (extract + match) and Step 3 (pick the card). **Step 2 is plain code.** |
| **Principle** | Simplicity first. It is a demo and a measurement, not a product. |

---

## Built: what it does and how it scored

**What runs, all locally:**
- **The pipeline:** `standup-exp board run` goes through all 24 stand-ups in chained mode. Each stand-up reads
  its board from the JSON database and commits back to it, and the run is scored against the answer key.
- **The static board:** `standup-exp board ui` builds it. Each stand-up has a **How it decided** panel: what was
  said, then the context cards pulled (with the match ticked), then the decision and its rule.
- **The live demo:** `standup-exp board demo` starts a local page at `localhost:8765`. Press **Run stand-up** and
  watch each step. The stand-up is committed to the database before the page is told it's done.
- **The dataset & trace explorer:** also built by `standup-exp board ui`. It has one row per answer-key item
  (plus every extra mention) showing raw text → extracted → matched → board context → decision → result, each
  checked right or wrong. Click a row to open its trace, laid out like a Foundry trace: a root span for the
  decision and a child span per step. Model calls show the system, user and assistant messages (with this row's
  part highlighted), the schema, tokens, latency and cost; code steps show their arguments and result.
- **The stand-in database:** `results/board/db-context-gpt-5.6-luna.json`, with one list per table: `standups`,
  `mentions`, `decisions`, `cards`, `card_history`. After the batch run it holds 24 stand-ups, 211 mentions,
  211 decisions, 80 cards and 188 history rows. Rows are flat, apart from two diagnostic id lists. Each
  decision row carries the model's one-sentence `pick_reason`.

Results in chained mode, GPT-5.6 Luna for every model step, 24 stand-ups:

| Metric | v0: whole board | **POC: board context** |
|---|---|---|
| Updates exactly right | 96% | **97%** |
| Updates linked to the right card | 98% | 97% |
| Wrong closes | 0 of 52 | 1 of 52 |
| Duplicate cards | 0 | 0 |
| New jobs with every field right | 67% | **67%** |
| Cards fully right after the last stand-up | 63% | **65%** |
| Cost per stand-up | $0.0053 | $0.0074 |

**Per step (new in the POC).** These show where any error comes from:

| Step | Score |
|---|---|
| 1, extract | found 100% of true new jobs and updates; `said` right 99% of the time |
| 2, context | the right card was in the list for 100% of updates, 4.4 cards on average instead of the whole board |
| 3, pick | picked the right card 97% of the time when it was in the list, with a one-sentence reason for every pick |
| Rules | 21 of 22 update remarks with no card were ignored, so no phantom cards were created |

**Reading it:** the POC matches or slightly beats v0 end to end, and every change can now be explained. It costs
about 40% more, because Step 1 now extracts updates as well as new jobs and Step 3 writes a reason for each pick.
The remaining errors sit in Step 3's pick and in new-job fields (owner, equipment, due date), not in finding the
right cards.

**Changed during the build.** Each came from reading the per-step scores:
1. **Step 2 uses the matcher's top-3 shortlist candidates** for names and equipment, not just its single pick.
   Unseen nicknames for the owner ("Haley", "Mide", "the boss") were being linked to the wrong person. This
   raised the share of updates with the right card in context from 92% to 100%.
2. **Step 1's prompt now says a job first raised in this meeting is always `new`,** even when it's handed from one
   person to another before someone accepts it. Those jobs were being labelled `reassigned` and then ignored.
3. **Fixes from the GPT-6 Astra reviews:**
   - a change is skipped only if it exactly repeats that card's previous change, so Thursday → Friday → Thursday
     ends on Thursday;
   - a restated job compares due date and time against the card *as this stand-up has changed it*;
   - the database is saved after every stand-up;
   - per-step scores credit a mention only if it's about the same job;
   - the live page rebuilds its steps panel from the database after a reload.
4. **Step 3 writes a one-sentence `reason` before it names the card.** The deployment returns no reasoning
   summary, so without this the trace could show what the model picked but not why. The reason comes first in
   the schema, so the model states it before it commits to a card. Pick accuracy went from 96% to 97%, for
   about $0.0002 more per stand-up.

## 1. In one paragraph

Every stand-up runs the same three steps:

1. **Step 1 · Extract what was said.** The existing extractor and matcher pull out every job mention: who
   spoke, who it's about, what the job is, which equipment, and what was said about it (new, done, progress,
   blocked, cancelled, reassigned or a new date).
2. **Step 2 · Pull board context.** Plain code asks the board for the open cards of the people involved, plus any
   card on the same equipment. That short list is what the mention could be about, and people can see it
   in the UI.
3. **Step 3 · Update or create.** For each mention, a model call picks the matching card from that short list, or
   none. Fixed rules then act on the answer:
   - a match → update that card;
   - said as new, with no match → create a card;
   - an update with no match → ignore it (it never creates a card).

The board that Step 3 writes is what Step 2 reads at the next stand-up. On the first stand-up the board is
empty, so everything said as new is created.

## 2. How we got here

| Stage | What it is | Where |
|---|---|---|
| Extraction experiment | Which method turns a transcript into dashboard-ready jobs? The best practical choice is GPT-5.6 Luna extracting → GPT-5.6 Luna matching. | [`README.md`](../../README.md), [`RESULTS.md`](../01-extraction-experiment/RESULTS.md) |
| Board loop v0 (built) | Step 1 adds new jobs. One "update check" call sees **the whole open board** plus the transcript, and returns the changes. 96% of updates exactly right, 98% on the right card, 0 of 52 wrong closes, $0.005 per stand-up. | `src/standup_exp/board/`, `docs/04-board-pipeline/mock-board/index.html` |
| Agent idea (explored) | A MAF agent calls `get_board()` and decides with tools. | [`docs/03-agent/DESIGN.md`](../03-agent/DESIGN.md) |
| **This design** | The same idea as a **pipeline**. Board context is pulled per person by code, and a model only picks within that short list. | this document |

**Why a pipeline rather than the agent:**
- **Predictable:** the same steps, in the same order, every time.
- **Testable step by step:** Step 2 is testable without any model at all.
- **Cheaper and easier to explain:** each change reads as "these were Tomasz's cards, and it picked #2".
- **Same UI:** the board page is unchanged.

**How it differs from v0:**
- v0 hands the model *everything*: the whole transcript and the whole open board.
- This design narrows the problem first:
  - Step 1 captures each mention;
  - Step 2 shrinks the board to the few cards that mention could be about;
  - Step 3 only has to choose among those cards, and it doesn't need the transcript at all, just the mention and its quote.

That scales with board size, and every decision can be explained.

**Evidence that the narrowing is safe.** We checked Step 2 on the existing answer key with no model: pull open cards
for whoever spoke, whoever was named, and anything on the same equipment. The right card was in the list for
**108 of 108** updates, with **2.6 cards on average** instead of the whole open board (**8.8**, up to 14).
This is a ceiling, because it assumes Step 1 extracted the mention correctly. §9 measures the real thing.

## 3. Goals and non-goals

**The POC must show:**
1. **The three steps working** on synthetic stand-ups: cards created on stand-up 1, then updated or created
   correctly at each stand-up after, with every change traceable to what was said and to the context that was pulled.
2. **A UI that shows it,** for each change: what was said, the cards pulled from the board, and the decision.
3. **Numbers:** the same end-to-end scores as v0 on the same 24 stand-ups, plus a score for each step, so we know
   which step to improve.

**Not in scope:** agents, deployment, authentication beyond `az login`, multiple users, approval flows, a real
task system (Planner, Azure DevOps, Jira), and other models.

## 4. The pipeline, step by step

### Step 1 · Extract what was said (model: the existing extractor + matcher)

This is the experiment's extractor, **widened** to keep updates, followed by the existing matcher, unchanged.

**Prompt change:** today's extractor is told *not* to extract finished, blocked or cancelled work. Here it
extracts that work too, and labels it with `said`. Chit-chat and plain observations are still dropped.

**Schema change:** the existing job fields, plus:

| Field | Meaning |
|---|---|
| `said` | `new`, `progress`, `done`, `blocked`, `cancelled`, `reassigned` or `due_changed` |
| `speaker` | who said it, a display name from the transcript, mapped exactly to a person id |
| `owner_mention` | whose job it is, *if named*: "Dave asked me to say…", "Marc, can you…" |
| `new_owner_mention` | reassigned only: who takes it over |
| `note` | a few words for the card's history ("waiting on the supplier") |
| `quote` | the transcript sentence, copied exactly |

**Matcher:** the existing `LLMNormaliser` turns the owner, equipment and site words into ids, handling unseen
nicknames ("the Atlas" → Air Compressor 1) exactly as in the experiment. The due date is resolved by the existing `dates.py`.

### Step 2 · Pull board context (code, no model)

For each mention, it asks the board which open cards the mention could be about:

```text
context(mention) = open cards where
      owner ∈ { speaker, named owner, named new owner }      # each: the matcher's pick + its top-3 shortlist
   or equipment is the same as, a part of, or the parent of the mention's equipment   # pick + top-3 shortlist
   or (said = reassigned and the card has no owner)          # "I'll pick up the one nobody had"
ranked by wording similarity, at most 8 cards
```

- **Which board:** the board as it was **at the start of the stand-up**, so the order of mentions doesn't matter.
- **What it returns:** a short list of `{card id, job, owner, equipment, status, due, latest note}`, which Step 3
  reads and the UI shows.
- **Cost and testing:** no model and no cost, so it can be tested exhaustively offline.

### Step 3 · Update or create (model picks, fixed rules act)

**One model call per stand-up.** For every mention with a non-empty context list, the model picks one card id
**from that mention's own list**, or `none`. The allowed ids are constrained in the JSON schema, and each pick is
checked against its own list. Mentions with an empty list skip the model entirely. Each pick is
`{mention, reason, card}`: the model writes a one-sentence reason before the card id, and the reason is saved on
the decision row so every match (or non-match) explains itself.

The model doesn't need the transcript: it gets each mention's job, `said`, speaker, owner, equipment and quote, plus
its context list (each card's job, owner, equipment, status and due date).
That makes the call smaller than v0's.

Then fixed rules decide the action:

| `said` | Matched a card? | Action |
|---|---|---|
| `new` | no | **create** a card (owner, equipment, site, due, priority, category) |
| `new` | yes | **comment** on that card: it was mentioned again, so no duplicate is made |
| progress / done / blocked / cancelled / reassigned / due_changed | yes | **update** that card: status, owner or due date, plus the note |
| progress / done / blocked / cancelled / reassigned / due_changed | no | **ignore**, logged as "update with no card". It never creates a card. |

**Write:** each action is a `BoardEvent`, applied with the existing `apply`. The board after the stand-up is the
input to the next stand-up's Step 2.

## 5. Walkthrough: stand-up 2 (as in the animation)

The board at the start holds #1 GT1 readings (Tomasz), #2 Fire alarm test (Tomasz) and #3 Book contractor (Hannah).

| What was said (Step 1) | Context pulled (Step 2) | Decision (Step 3) |
|---|---|---|
| Tomasz · fire alarm test · **said: done** | Tomasz's cards: #1, #2 | #2 matches → **update: done** |
| Tomasz · GT1 readings · **said: progress** | Tomasz's cards + GT1: #1, #2 | #1 matches → **update: in progress**, "Made a start" |
| Hannah · contractor · **said: blocked** | Hannah's cards + GT2 valve: #3 | #3 matches → **update: blocked**, "Waiting on supplier" |
| Marcus · order parts, the Atlas → Air Compressor 1 · **said: new** | none | no match, said new → **create #4** |
| Grace · long belt · **said: done** | none | no match, not new → **ignore** |

Result: 3 updated, 1 created and 1 ignored. Stand-up 3's Step 2 reads this board.

## 6. How it changes the UI

**Rule: only Step 3's write can change the board, and the page only displays it.** Every step's output is
recorded, so the page can explain each change.

Every recorded item:

| Item | Comes from | Shown as |
|---|---|---|
| `said` | Step 1 | the mention list, and the highlighted transcript line |
| `context` | Step 2 | the cards pulled for that mention, also outlined on the board |
| `decision` | Step 3 | the match (or none) and the rule applied |
| `card` | the write | the card created, moved or updated |

There are two ways to show it, and the first needs no new infrastructure:

1. **Report mode (build first).** The existing static board page (`docs/04-board-pipeline/mock-board/index.html`) gains a **"How it
   decided"** panel. Click any change to see what was said → the cards pulled → the decision and the rule. It is
   built from the results file by `standup-exp board ui`, as today.
2. **Live mode (optional, local).** `standup-exp board demo` starts a small local FastAPI app. Press **Run stand-up**
   and the items above stream to the page as each step finishes: the mention appears, the context cards light up,
   then the card moves. The mechanism is the one already researched for the agent design: each stand-up is
   committed to the JSON stand-in database, then each item is pushed to the page over Server-Sent Events. No agent
   and no extra protocol are needed.

## 7. Data

- **Reuse `data/series` as-is:** 3 teams × 8 consecutive stand-ups, 81 new jobs, 108 updates.
  - Each stand-up has an answer key. A simulator plans each card's fate, GPT-6 Astra writes the meeting, and a
    validator checks everything is actually said.
  - The hard cases are tagged: updates for a colleague or someone who is off, two jobs on the same equipment,
    activity-only references, unseen nicknames, and "done" remarks with no card.
- **More data if needed:** `standup-exp board generate --series N --days D --writer gpt-6-astra` (about $0.13 per stand-up).
- **The pipeline never sees the answer key.** It gets only what a real system would: the transcript, the date, the
  roster and its own board.

## 8. What is reused and what is new

| Reused as-is | New |
|---|---|
| `data/series` and the simulator/writer | Step 1: the widened extractor schema and prompt (`said`, speaker, note) |
| `LLMNormaliser` (owner, equipment, site ids), `dates.py` | Step 2: `context_for(mention, board)` (plain code) |
| `BoardEvent`, `apply`, chained mode | Step 3: the pick call plus the rules table |
| `board.evaluate` scorer, `mock-board` page | Per-step scores; the "How it decided" panel |

## 9. How it is measured

**End to end:** the same scorer and table as v0, on the same 24 stand-ups, in chained mode.

| Metric | v0: whole board (built) | POC: board context (built) |
|---|---|---|
| Updates exactly right | 96% | 97% |
| Updates linked to the right card | 98% | 97% |
| Wrong closes | 0 of 52 | 1 of 52 |
| Duplicate cards | 0 | 0 |
| New jobs with every field right | 67% | 67% |
| Cards fully right after the last stand-up | 63% | 65% |
| Cost per stand-up | $0.0053 | $0.0074 |

**Per step, so we know what to fix:**

| Step | Score | How |
|---|---|---|
| 1 | Mention recall: share of true new jobs and updates extracted; `said` correct | Match mentions to the answer key by quote and wording (the existing matcher logic) |
| 2 | Context recall: right card in the list; average list size | Offline with the answer key's mentions (**108/108, 2.6 cards**), then with Step 1's real mentions |
| 3 | Pick accuracy when the right card *is* in the list; wrong closes; duplicates; "done" remarks correctly ignored | From the recorded decisions |

**The POC succeeds if** (set before the build; the outcome is in brackets):
1. it runs end to end on all 24 stand-ups and the UI explains every change (met);
2. it is not clearly worse than v0 on updates and wrong closes, with 108 updates the margin of error is about ±4 points (met: 97% vs 96%, 1 wrong close vs 0);
3. it costs no more than v0 (missed: about 40% more, because Step 1 now extracts updates too and Step 3 writes a reason; still under a cent a stand-up).

It is a clear *win* if Step 3 is cheaper than v0's update check and the per-step scores show where the remaining errors come from (met).

## 10. Build plan

Each milestone ends with something checkable.

| # | Milestone | Model calls | Done when |
|---|---|---|---|
| M0 | **Step 2 as code** + offline tests | none | It reproduces 108/108 context recall on the answer key; the list-size cap is tested |
| M1 | **Widen Step 1** (schema + prompt) | 1 run over 24 stand-ups (cents) | Mention recall and `said` accuracy reported |
| M2 | **Step 3** pick call + rules + write | offline tests use the answer key's mentions | With perfect mentions, the board matches the answer key |
| M3 | **End-to-end chained run**, scored against v0 | 1 run (cents) | The table in §9 filled in, plus the per-step scores |
| M4 | **"How it decided" panel** in `mock-board/` | none | Every change on the board explains itself; the headless-browser check passes |
| M5 | Live local demo (optional) | as M3 | `standup-exp board demo` streams each stand-up step by step |

## 11. Risks

| Risk | Mitigation |
|---|---|
| Step 1 now extracts more (updates too), so more noise | Rules only *create* from `said: new`; updates with no match are ignored. Measured by mention recall and duplicates. |
| The owner isn't named and the speaker isn't the owner (a colleague reporting) | Step 2 also pulls cards on the same equipment. Tagged `third_party` in the breakdown. |
| Activity-only references with no equipment ("the readings") | The person's own cards cover it; wording similarity ranks them. Tagged `activity_only`. |
| Two open jobs on the same equipment | Both are in context; Step 3 picks by the job itself. Tagged `twin` (v0's weakest, at 90%). |
| Context lists grow on big boards | Capped at 8, ranked by wording similarity; context recall reports it. |

## 12. Layout and commands

```
src/standup_exp/board/
  context.py           Step 2: context_for() (plain code)
  context_pipeline.py  Step 1 (mentions + existing matcher), Step 3 (pick), the rules, run_day / run_series
  store.py             the stand-in database: one JSON file, a list per table, atomic save after every stand-up
  demo.py              live local demo: FastAPI, one guarded POST per stand-up, SSE stream of each step
  pipeline.py          v0 (whole board), kept for comparison
  evaluate.py          shared scorer + per-step scores
  explorer.py          builds the dataset & trace explorer from the results file (no model calls)
docs/04-board-pipeline/
  mock-board/          static board with the "How it decided" panel (template.html -> index.html)
  trace-explorer/      dataset & trace explorer (template.html -> index.html)
  live-demo/index.html the live demo page (served by demo.py)
results/board/
  db-context-gpt-5.6-luna.json      the database after the batch run
  db-demo-gpt-5.6-luna.json         the live demo's database (Reset team clears it)
  latest-context-gpt-5.6-luna.json  scores + per-stand-up records (with model-call traces) for both static pages
tests/test_board.py    offline: context rules, rules, database round trip, perfect-steps oracle, negative cases, traces
```

```powershell
standup-exp board run --pipeline context --model gpt-5.6-luna   # POC, chained, scored, writes the database
standup-exp board run --pipeline whole-board                     # v0, for comparison
standup-exp board ui                                             # static board + dataset & trace explorer
uv pip install -e ".[demo]"                                      # fastapi + uvicorn, for the live demo
standup-exp board demo --open                                    # http://127.0.0.1:8765
```

## 13. Decisions

| Decision | Choice | Why |
|---|---|---|
| Agent or pipeline | **Pipeline** | Predictable, testable per step, cheaper, easy to explain |
| Where board context comes from | **Code, by person and equipment** | No model needed; checked at 108/108 on the answer key |
| Who picks the card | **A model, only within the context list** | Handles wording and nicknames; can't pick outside the list |
| What creates cards | **Only `said: new` with no match** | Stops "done" remarks with no card from creating cards |
| Board for context | **As at the start of the stand-up** | The order of mentions doesn't matter |
| Context keys | **The matcher's pick plus its top-3 shortlist candidates** | Unseen nicknames were mis-linked; Step 3 stays precise |
| Restated job (`new` + match) | **Comment, plus a new due date or owner if one is given** | Your choice; no duplicate card |
| Storage | **One JSON file, a list per table**: `standups`, `mentions`, `decisions`, `cards`, `card_history` | Stands in for the real project's database; flat rows load into Power BI |
| UI | **The "How it decided" panel and a live local demo** | Your choice; both read what the pipeline recorded |
| Models | **GPT-5.6 Luna for Steps 1 and 3**; OpenAI embeddings in the shortlist; Step 2 has no model | The experiment's best pipeline, and like for like with v0 |
| Hosting | **Local only** | It's a POC demo |

## 14. Open questions

1. Should context also include cards closed at the last stand-up, so corrections like "actually that isn't done yet"
   can reopen them? **Not in the POC.** The synthetic data has no reopen cases.
2. ~~When a `said: new` mention matches an existing card, should it also update a changed due date or owner?~~
   **Decided: yes.** It comments, and updates the due date or owner when the restatement gives new ones.
3. ~~One Step 3 call per stand-up, or one per mention?~~ **Built as one call per stand-up.** Each pick is checked
   against that mention's own list.
4. **New:** is the 36% extra cost worth the explainability? One cheaper option is to extract updates only when the
   board has open cards. That's for after the POC.
