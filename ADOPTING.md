# Using this for your own problem

Take what's useful. You don't need all of it. Most people will want the board pipeline (approach 4) and the
lessons in [`FINDINGS.md`](FINDINGS.md). The big model experiment (approach 1) is there if you want to re-test
models on your own data.

## 1. Look first, no install

These are single HTML files. Double-click them, or paste the path into a browser.

| Page | What you'll see |
|---|---|
| [`docs/04-board-pipeline/mock-board/index.html`](docs/04-board-pipeline/mock-board/index.html) | The board after each of 24 stand-ups, with "How it decided" for every change and a compare-with-the-answer-key mode |
| [`docs/04-board-pipeline/trace-explorer/index.html`](docs/04-board-pipeline/trace-explorer/index.html) | Every decision as a row, right or wrong, and its full trace: prompt, input, output, the model's reason, the code steps |
| [`docs/02-walkthrough/v2-report/index.html`](docs/02-walkthrough/v2-report/index.html) | The model experiment: results with margins of error, and every meeting, prompt and result |
| [`docs/02-walkthrough/v1-deck/index.html`](docs/02-walkthrough/v1-deck/index.html) | The same experiment as a 19-slide deck |

The live demo (`live-demo/index.html`) is the one page that needs the local server. See step 7.

## 2. What needs what

| You want to... | Needs |
|---|---|
| Open the pages above | Nothing |
| `pytest -q`, `standup-exp demo`, `show`, `board show`, `board ui`, `report`, `walkthrough --reuse` | Python only |
| Make simple test stand-ups: `standup-exp board generate --renderer template` | Python only |
| Run the pipeline (`board run`), the live demo (`board demo`), or write realistic stand-ups (`board generate`, `generate`) | Your model deployments, **or** our reply cache |

**Our reply cache.** Every model reply we got is saved in `results/cache/` (33 MB). It isn't in git. If you get a
copy from us and drop it in that folder, `standup-exp board run --cache-only` and the live demo replay our runs
exactly, for free, with no Azure at all:

```powershell
$env:LLM_CACHE_ONLY = "1"; standup-exp board demo --open
```

## 3. Install (about 5 minutes)

Clone the repo and install it **editable** (`-e`). It reads `data/` and `docs/` from the folder you cloned, so a
plain `pip install .` won't work.

```powershell
cd standup-extraction-experiment
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,demo]"   # enough for the board pipeline, the tests and the offline demo
.venv\Scripts\Activate.ps1                                 # macOS / Linux: source .venv/bin/activate

pytest -q          # offline tests, no models
standup-exp demo   # offline end to end with the non-model methods
```

Add `local,train` to the extras (`".[local,train,dev,demo]"`) only if you want the GLiNER and local-embedding methods
from the model experiment. They're big, and they didn't win.

## 4. Point it at your models

1. Copy `.env.example` to `.env` and fill in the endpoint URLs. Keys are optional: `az login` is used when a key is
   empty.
2. In [`data/models.yaml`](data/models.yaml), change each `deployment` to your own deployment name. You only need
   the models you'll use. The board pipeline uses one model for every step (`--model`, default `gpt-5.6-luna`),
   plus the embedding model at the bottom of the file for the name and machine shortlists.
3. Try a tiny run first. This makes 2 simple stand-ups for free, then runs the pipeline on them (a few model calls):

   ```powershell
   standup-exp board generate --renderer template --series 1 --days 2 --out data\series-try
   standup-exp board run --model gpt-5.6-luna --data data\series-try
   ```

## 5. Swap in your world

Everything reads from [`data/ontology.yaml`](data/ontology.yaml): the synthetic data, the prompts, the shortlists
and the scorer.

- **People**, with the nicknames they get called.
- **Sites**, for off-site work.
- **Assets** as a hierarchy (line → machine → part), with nicknames. Put some nicknames under `heldout_aliases`
  so they only show up in test data. That's how you test the matcher on words it has never seen.
- **Categories and priorities**, with the rubric the models follow.

[`DATA.md`](DATA.md) explains the ontology in more detail, how we generated the data, and which changes improved the
results.

Two things are fixed in code as well, so keep them or change both places:
- the three category names (`operations`, `safety`, `organisation`) and the 1–3 priority scale are also in the
  output schemas in `board/context_pipeline.py`, `extractors/llm.py` and `normalisers/llm.py`;
- a few rubric lines are added in `ontology.py` (`rubric()`), and the board prompt reads the ontology once, when the
  module loads.

Then make stand-ups for your world:

```powershell
standup-exp board generate --writer gpt-6-astra --out data\series-mine      # realistic, about $0.13 a stand-up
standup-exp board generate --renderer template --out data\series-mine2      # free, simpler, good for a smoke test
standup-exp board show s01d03 --data data\series-mine                       # read one, with its answer key
```

Use a writer model that you **aren't** testing. See "writer bias" in [`FINDINGS.md`](FINDINGS.md).

## 6. Pick your model (optional)

Only if you want to re-test models on your own meetings. Name the models you have, not `all`:

```powershell
standup-exp generate --writer gpt-6-astra --n 20 --out data\meetings-mine
standup-exp run --extractors llm --normalisers llm --models gpt-5.6-luna,gpt-6-luna --data data\meetings-mine
standup-exp report
```

## 7. Run the board pipeline on your data

```powershell
standup-exp board run --model gpt-5.6-luna --data data\series-mine     # scored, writes the JSON database
standup-exp board ui --model gpt-5.6-luna --data data\series-mine      # rebuilds the board and trace explorer pages
standup-exp board demo --model gpt-5.6-luna --data data\series-mine --open   # live demo at http://127.0.0.1:8765
```

## What to lift into your own code

| Piece | File | Lines | What it gives you |
|---|---|---|---|
| Pull the cards (Step 2) | [`board/context.py`](src/standup_exp/board/context.py) | 38 | That person's open cards and cards on the same machine (or its parent or parts), ranked, at most 8 |
| Extract, pick and the rules (Steps 1 and 3) | [`board/context_pipeline.py`](src/standup_exp/board/context_pipeline.py) | 310 | The prompts, strict schemas, the rules table, and the loop where each stand-up starts from the last board |
| Cards and changes | [`board/schema.py`](src/standup_exp/board/schema.py) | 172 | Card and event models, and `apply()` to play events onto a board |
| The stand-in database | [`board/store.py`](src/standup_exp/board/store.py) | 130 | Five flat tables. Swap it for yours (below) |
| Model client | [`llm.py`](src/standup_exp/llm.py) | 354 | Strict JSON schema output, a disk cache, cost per call, and `trace_calls()` for traces |
| Scorer | [`board/evaluate.py`](src/standup_exp/board/evaluate.py) | 341 | End-to-end and per-step scores against an answer key |
| Trace explorer | [`board/explorer.py`](src/standup_exp/board/explorer.py) + [`template.html`](docs/04-board-pipeline/trace-explorer/template.html) | 324 | The Foundry-style trace page |

## Swapping the JSON file for your database

The pipeline only needs two things from a store:
- `board(series_id)` returns the team's current cards;
- `commit(day, board_after, why, standup_row)` saves the stand-up, its mentions and decisions, the cards and their
  history.

The five tables (`standups`, `mentions`, `decisions`, `cards`, `card_history`) map one to one onto database tables.
Each row is flat, and each decision row carries the model's `pick_reason`. The live demo also uses `standups_run`,
`reset`, `to_dict` and `meta`. The store is created in `board/runner.py` and `board/demo.py`, so that's where you'd
swap it.

## What you can skip

- GLiNER and fine-tuning: they didn't beat plain rules.
- The full 163-pipeline matrix, unless you want to re-test lots of models.
- The v1 slide deck: the v2 report shows the same thing.
- The agent design, unless your problem really needs one (see [`FINDINGS.md`](FINDINGS.md), section 4).
