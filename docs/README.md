# Docs: one folder per approach

The repo is one project with four approaches to the same problem: turning factory stand-up transcripts into jobs
on a board. Each approach has its own folder here, numbered in reading order. For what we learned across all four,
read [`FINDINGS.md`](../FINDINGS.md); for the ontology, how the data was made and what improved results, read
[`DATA.md`](../DATA.md); to reuse it, read [`ADOPTING.md`](../ADOPTING.md).

| # | Approach | Question | Status | Start here |
|---|---|---|---|---|
| 1 | [**Extraction experiment**](01-extraction-experiment/) | Which method turns a transcript into dashboard-ready jobs, and at what cost and speed? | Done | [`README.md`](01-extraction-experiment/README.md) · [`RESULTS.md`](01-extraction-experiment/RESULTS.md) |
| 2 | [**Walkthrough**](02-walkthrough/) | How do we present the experiment so people can check it? | Done: slide deck (v1), report (v2) | [`v2-report/index.html`](02-walkthrough/v2-report/index.html) |
| 3 | [**Agent**](03-agent/) | Can a MAF agent with board tools decide what to create and update? | Designed, then parked: code can load the board, so it doesn't need an agent | [`DESIGN.md`](03-agent/DESIGN.md) |
| 4 | [**Board pipeline**](04-board-pipeline/) | Once jobs are extracted, can each stand-up keep the board up to date? | v0 and the POC built (JSON database, live demo, dataset & trace explorer) | [`README.md`](04-board-pipeline/README.md) |

## What belongs to which approach

The code, data and results stay in the repo's top-level folders, because the approaches share the ontology, the
model catalogue, the model client and its reply cache.

| # | Docs (this folder) | Code (`src/standup_exp/`) | Data (`data/`) | Results (`results/`) |
|---|---|---|---|---|
| 1 | `01-extraction-experiment/`: `README.md` (method, data, scoring), `RESULTS.md`, hero image | `generate/`, `extractors/`, `normalisers/`, `runner.py`, `evaluate.py`, `report.py`, `finetune.py` | `meetings/`, `meetings-astra/` (and the offline `meetings-demo/`) | `latest-e2e-*.json`, `latest-normalise-*.json`, `runs/` |
| 2 | `02-walkthrough/`: `v1-deck/`, `v2-report/` | `walkthrough.py` (v1), `v2-report/build.py` (v2) | none; reads approach 1's results | `walkthrough-data.json` |
| 3 | `03-agent/`: `DESIGN.md`, `agent-loop.svg` | none (parked) | would have reused `series/` | none |
| 4 | `04-board-pipeline/`: `README.md`, `DESIGN.md`, `board-loop.svg`, `mock-board/`, `trace-explorer/`, `live-demo/` | `board/` | `series/` | `board/` (including the JSON databases) |
| shared | `findings/`: the animations and screenshots in `FINDINGS.md` and `DATA.md` | `llm.py`, `dates.py`, `ontology.py`, `schema.py`, `paths.py`, `cli.py` | `ontology.yaml`, `models.yaml` | `cache/` (every model reply and embedding; not in git) |

## Rebuilding the HTML files

None of these call a model. They re-render from files that already exist.

```powershell
standup-exp report                                  # 01-extraction-experiment/RESULTS.md
standup-exp walkthrough --reuse                     # 02-walkthrough/v1-deck/index.html (reuses the deck's own data)
python docs\02-walkthrough\v2-report\build.py       # 02-walkthrough/v2-report/index.html
standup-exp board ui                                # 04-board-pipeline/mock-board/ and trace-explorer/index.html
```
