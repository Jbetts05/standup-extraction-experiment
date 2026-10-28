# Approach 4 · Board pipeline: add cards, then keep them up to date

The extraction experiment (approach 1) stops once the jobs are extracted. This approach asks what happens
*after* extraction: the jobs go on a Kanban board, and each new stand-up must update the cards that are already
there ("finished that yesterday", "still waiting on parts", "Dave's taking it over"). Only the primary pipeline
from approach 1 (GPT-5.6 Luna) is used.

| | |
|---|---|
| **POC** | **Built as a local POC.** Board context is pulled per person, and a model picks the card from that short list and says why. It writes to a JSON stand-in database, and has a live demo and a trace explorer. See [`DESIGN.md`](DESIGN.md). |
| **v0** | **Built and scored first, kept for comparison.** Step 1 extracts new jobs, then one update check runs over the whole board. |

New here? [`FINDINGS.md`](../../FINDINGS.md) has the lessons in five minutes, and [`ADOPTING.md`](../../ADOPTING.md)
shows how to run it on your own data.

## POC (built): extract → pull board context → update or create

<p align="center">
  <img src="board-loop.svg" alt="Stand-up 1: Step 1 extracts three new jobs, Step 2 finds nothing on the empty board, and Step 3 creates three cards. Stand-up 2: for each thing said, Step 2 pulls that person's open cards and any cards on the same equipment, and Step 3 updates the matching card, creates a card for a new job with no match, or ignores an update with no match." width="100%">
</p>

Every stand-up runs three steps:
1. **Extract every mention** with GPT-5.6 Luna, new jobs and updates alike, then link people and equipment
   with the existing matcher.
2. **Pull that person's open cards,** and cards on the same equipment, from the board. This is plain code.
3. **Pick the matching card** from that short list with GPT-5.6 Luna. Fixed rules then update the card, create
   one (only for something said as new with no match), or ignore the mention.

The pipeline reads the board from, and writes it to, a **JSON file that stands in for the real project's
database**: one list per table (`standups`, `mentions`, `decisions`, `cards`, `card_history`), with flat rows
that load into Power BI.

| Chained, GPT-5.6 Luna | v0 | POC |
|---|---|---|
| Updates exactly right | 96% | **97%** |
| Wrong closes | 0 of 52 | 1 of 52 |
| New jobs with every field right | 67% | **67%** |
| Cards fully right after the last stand-up | 63% | **65%** |
| Cost per stand-up | $0.0053 | $0.0074 |

Per step:
- **Step 1** found 100% of new jobs and updates.
- **Step 2** put the right card in context 100% of the time, with 4.4 cards on average.
- **Step 3** picked right 97% of the time, and gives a one-sentence reason for every pick.
- **The rules** ignored 21 of 22 update remarks that had no card.

```powershell
standup-exp board run --pipeline context --model gpt-5.6-luna   # chained, scored, writes results/board/db-context-*.json
standup-exp board ui                                             # static board ("How it decided" panel) + dataset & trace explorer
uv pip install -e ".[demo]"                                      # fastapi + uvicorn
standup-exp board demo --open                                    # live local demo at http://127.0.0.1:8765
```

**Live demo.** Pick a team and press **Run stand-up**. For each thing said, the page plays four steps:
1. its line in the transcript is highlighted (the rest is dimmed);
2. an extraction box shows what Step 1 pulled out: the job, who said it, the owner, the equipment ("the Atlas" →
   Air Compressor 1), the due date and a note, and whether it's new or an update;
3. the cards pulled from the board are listed and outlined on the board;
4. the match is ticked, the model's reason for its pick and the rule are shown, and the card changes.

**Speed** can be *Step by step* (each step waits for **Next** or the → key), slow, normal or fast, and **Pause**
stops playback at any point. Once a stand-up finishes, click any line in Step 1 to review how it was handled.

The stand-up is committed to `results/board/db-demo-gpt-5.6-luna.json` before the page is told it's done.
Reloading the page never re-runs anything. **Reset team** starts that team again. Model replies are cached, so
replaying the synthetic stand-ups costs nothing.

**Dataset & trace explorer** ([`trace-explorer/index.html`](trace-explorer/index.html), linked from the board
header) is the technical view, set out like an experiment-tracking tool:
- **Dataset:** one row per answer-key item (189: 81 new jobs, 108 updates) plus the 22 extra mentions the
  pipeline made. Each row shows raw text → extracted → matched → decision, and whether it's right; if it's
  wrong, the row shows the stage where it went wrong. Filter by team, stand-up, type, result or failing stage,
  or click a summary tile (e.g. "27 wrong at Step 1 · match").
- **Trace:** click a row to open its trace, laid out like a trace in Microsoft Foundry. On the left there's a
  root span for the decision, with a child span for every step and a timeline bar showing where the time went:
  1. extract (model);
  2. shortlist (code);
  3. link to master data (model);
  4. pull board context (code);
  5. pick the card (model);
  6. rules (code);
  7. write to the database;
  8. evaluate.

  Step through with ← → (↑ ↓ changes row). Each span has three tabs:
  - **Input & output.** For a model call, the **system** message (the prompt), the **user** message (the
    transcript or JSON it was given) and the **assistant** message (its JSON reply), with this row's part
    highlighted in each, plus the strict response schema. A box at the top says what the model decided for
    this row. For a code step, its arguments and result.
  - **Attributes.** Model, deployment, tokens, latency, cost and cache key, using the OpenTelemetry GenAI names
    that Foundry traces use (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, …).
  - **Raw span.** The whole span as JSON.

  The root span tells the story in one place: what was said, what each step did, the model's reason and the rule
  applied, and the outcome. For a wrong field, the trace shows whether the right id was on the shortlist (so the
  shortlist missed it) or the matcher chose wrong. The URL keeps the row and span (`#row=…&span=…`), so you can
  link to one.
- **Why the model picked a card.** GPT-5.6 Luna reasons before it answers, but this deployment returns no
  reasoning summary (tested with `auto`, `concise` and `detailed`). So Step 3's schema asks for a one-sentence
  `reason` *before* the card id, e.g. *"Neither near-miss write-up card identifies which location this update
  concerns."* The reason is saved on the decision row in the database and shown in the trace, the live demo and
  the board's "How it decided" panel. Adding it cost about $0.0002 per stand-up and got one more update right.
- **What it shows:** every new-job error is in Step 1's field matching (mostly priority and equipment), and every
  update error is in Step 3's pick. Step 2 never lost the right card.

The traces are recorded when the pipeline runs (`llm.trace_calls()`), including replies from the cache, so
`standup-exp board run --pipeline context --cache-only` rebuilds them for free, as long as you have the reply cache
(`results/cache/`). The cache isn't in git; without it that command needs working model deployments.

## v0 (built first): extract, then one update check over the whole board

```
transcript --1. existing extractor + matcher (unchanged)--> new jobs --------------------+
     |                                                                                     v
     +--2. update check: open cards + transcript + new jobs --> updates, merges --> apply --> board
                                                                                           |
                                          tomorrow's stand-up starts from this board <------+
```

- **Step 1** is `llm@gpt-5.6-luna` → `llm@gpt-5.6-luna`, exactly as in approach 1.
- **Step 2** is one call per stand-up. It returns done, progress, blocked, cancelled, reassigned or
  due-date changes for open cards, each with a note and a quote for the card's history. It also flags
  new jobs that just restate an open card; these become a comment instead of a duplicate card. Card
  and person ids are enums in the strict JSON schema, so the model cannot invent them.
- **Data:** `data/series` holds 3 teams × 8 consecutive working days (24 stand-ups, written by
  GPT-6 Astra). A simulator decides each card's fate first, then the writer produces the meeting. Hard
  cases are tagged:
  - updates on a colleague's behalf, or for someone who is off;
  - two open jobs on the same equipment;
  - updates that name only the activity;
  - unseen nicknames;
  - "done" remarks about equipment that has no card.
- **Scored in chained mode**, where tomorrow starts from the pipeline's own board, so mistakes carry forward.

| GPT-5.6 Luna, every step | Result |
|---|---|
| Updates exactly right (card, change, new owner or date) | 96% of 108 |
| Updates linked to the right card | 98% |
| Wrong closes (done or cancelled on a card that wasn't) | 0 of 52 |
| New jobs with every field right (step 1) | 67% |
| Cards fully right after the last stand-up | 63% |
| Cost | $0.0053 per stand-up |

The update loop is reliable. The board's weak spot is step 1's field accuracy, because a wrong owner or
equipment on a new card stays wrong.

```powershell
standup-exp board generate --writer gpt-6-astra   # 3 teams x 8 days into data/series
standup-exp board show s02d05                     # one stand-up: open cards, transcript, answer key
standup-exp board run --pipeline whole-board --model gpt-5.6-luna   # chained; add --modes chained,isolated for the diagnostic
standup-exp board ui --pipeline whole-board                          # mock-board/index.html for v0 (no model calls)
```

The mock board ([`mock-board/index.html`](mock-board/index.html)) is a single self-contained file:
- **Board:** To do / In progress / Blocked / Done columns with the owner on every card, a person
  filter and a stand-up picker.
- **Stand-up panel:** the changes each stand-up made, plus the transcript.
- **Compare with answer key:** marks each card right or wrong, and shows missed cards and false closes.
- **Light/dark toggle** in the header.

## Where things live

| | Path |
|---|---|
| Code | `src/standup_exp/board/`: POC in `context.py` (Step 2), `context_pipeline.py` (Steps 1 and 3 + rules), `store.py` (JSON database), `demo.py` (live demo), `explorer.py` (dataset & trace explorer); v0 in `pipeline.py`; shared `simulate`, `render`, `evaluate`, `export`, `cli` |
| Tests | `tests/test_board.py` (offline) |
| Data | `data/series/`: one `.json` (answer key) + `.vtt` per stand-up |
| Results | `results/board/latest-<pipeline>-<model>.json` (scores and records), `db-context-<model>.json` (the database), `db-demo-<model>.json` (the live demo's database) |
| UI | [`mock-board/`](mock-board/) and [`trace-explorer/`](trace-explorer/): `template.html` → `index.html` (both built by `standup-exp board ui`); [`live-demo/index.html`](live-demo/index.html) (served by `standup-exp board demo`) |
| Design | [`DESIGN.md`](DESIGN.md), [`board-loop.svg`](board-loop.svg) |
