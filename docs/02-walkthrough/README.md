# Approach 2 · Walkthrough

Two ways of presenting the extraction experiment (approach 1). Both are single self-contained HTML files built
from the same data export, and neither calls a model.

| Version | What it is | Rebuild |
|---|---|---|
| [`v2-report/`](v2-report/index.html) | A tabbed report: summary, results with margins of error, compare, a data explorer with a transcript drawer, and the method. [`PROPOSAL.md`](v2-report/PROPOSAL.md) explains why it was restructured. | `python docs\02-walkthrough\v2-report\build.py` |
| [`v1-deck/`](v1-deck/index.html) | The original 19-slide deck | `standup-exp walkthrough --reuse` |
