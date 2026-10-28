# Walkthrough v2: proposed structure for the experiment report

`index.html` in this folder is a working prototype of the recommendations below. It is built from the
same data export as the current deck (`docs/02-walkthrough/v1-deck/`), which is left unchanged.

```powershell
python docs\02-walkthrough\v2-report\build.py              # reads results\walkthrough-data.json
python docs\02-walkthrough\v2-report\build.py --from-index # or re-uses the payload inside docs\02-walkthrough\v1-deck\index.html
.venv\Scripts\python docs\02-walkthrough\v2-report\build.py --fresh  # or exports the latest results\latest-*.json directly (read-only)
```

## 1. Problems with the current deck

| Area | Problem | Evidence in `docs/02-walkthrough/v1-deck/template.html` |
|---|---|---|
| Story | The answer comes last. Verdict cards first appear on slide 13 of 19 and takeaways on slide 19. Slide 1 opens with a 25-row component inventory. | `slide("Overview", "The question")` renders `inventory()` |
| Story | One linear deck serves three audiences: decision-maker, reviewer and practitioner. | 19 slides in sequence, and every audience must page through all of them |
| Story | No hypothesis, success threshold or ship bar. Nothing says what score would be good enough. | no threshold anywhere |
| Story | No primary result. Two writer sets are shown one at a time, each with a different winner. | `setControl()` on every results slide |
| Data | Scores are absolute only. Nothing is compared against the baseline, and there is no gap to the best. | Leaderboard, Foundry models |
| Data | Ranked 1..N even though most rows are within the ± margin of each other. | `Leaderboard:all-pipelines` table |
| Data | No field-level view across pipelines, so you can't see *why* jobs fail to be dashboard-ready. | per-field counts exist only for one pipeline on the Match slide |
| Data | No summary of failure types (missed, not real, wrong field). | – |
| UX | Hidden shared state. Extractor, matcher, meeting and test set persist across slides, so the same slide changes meaning. | `S.ex`, `S.no`, `S.pipeMeeting`, "Scope ·" chips |
| UX | Raw data, extraction, matching and scoring are split over 7 slides (04–07, 09–11) for the same meeting. | Datasets → Answer key → Transcript → Validation → Extract → Match → Score |
| UX | Views can't be linked. Only the slide number is in the URL. | `history.replaceState(null, "", "#" + n)` |
| Method | Margins treat each job as independent (`1.96·√(p(1−p)/n)` with n = jobs), but jobs cluster inside 8 meetings. | `moe()` |
| Method | "Differences smaller than the margin are ties" compares two unpaired margins. Every pipeline is scored on the same jobs, so a **paired** test is the right one. | slide 02 text |
| Method | The matrix is incomplete by design, but pipelines that weren't run simply don't appear. | "Matrix rule" callout |
| Integrity | The narrative is hard-coded and has drifted from the data. The text says "11 LLMs"; this export scores 2 (GPT-5.4, Mistral Large 3) across 18 pipelines. Only the GPT-5.4-written set is scored, but the text describes both. | `LLM_LIST`, the "11 + 5" stat card |

## 2. Patterns from real experimentation platforms

| Platform | Pattern | Used in v2 as |
|---|---|---|
| Eppo, Statsig | Experiment page with Summary / Results / Explore / Design tabs, decision first | Six tabs, Summary first |
| Microsoft ExP, Optimizely | Scorecard: one row per variant, change vs control with CI, clear significance marking | Scorecard with **Δ vs baseline** and a forest plot per row |
| LMSYS Arena, HELM | Rank with CI; **tied groups** instead of 1..N; comparisons by scenario | *Close to #1 / Behind #1, beats regex / No better than regex / Worse than regex* verdicts; nickname and writer slices |
| Braintrust, LangSmith, promptfoo | Side-by-side comparison of two runs, filtered to examples that changed | **Compare** tab: 2×2 agreement table, paired gap, list of jobs where the two disagree |
| MLflow, W&B Reports | Run metadata, parameters, reproducibility; prose with embedded charts | **Appendix** tab: models, prices, prompts, run IDs |
| Experiment design docs | Hypothesis, primary / secondary / guardrail metrics, threats to validity | **Method** tab |

## 3. New structure

```
[ Summary ] [ Results ] [ Compare ] [ Data explorer ] [ Method ] [ Appendix ]
 Scope: <test set / pooled> · N meetings · N jobs · N pipelines · metric definition   ← always visible
```

1. **Summary** (decision memo, one screen)
   - Recommendation: the **highest-scoring** pipeline, with its yearly running cost.
   - "If cost matters" table: the cost–quality frontier, showing for each cheaper option the points given up (paired interval), dollars saved per year and extra jobs a year needing a human fix.
   - Headline number with its interval and gain over the baseline.
   - An adjustable **ship bar**. The bar is an explicit assumption, and the page states whether any pipeline clears it.
   - "What the evidence supports" bullets and caveats, all generated from the data so they can't drift.
2. **Results** (scorecard)
   - Group, change vs baseline with interval, F1 / precision / recall, cost and latency, sortable.
   - Cost-vs-quality chart with intervals, baseline and ship-bar lines.
   - **Where jobs are lost**: an outcome bar per pipeline (ready / 1 field wrong / 2+ wrong / missed) and a field heatmap.
   - Matching only: known vs unseen nicknames.
   - Writer robustness check (needs both writer sets scored).
3. **Compare**: pick A and B. Shows the paired gap with interval, a 2×2 agreement table, field-by-field gaps, and the jobs that changed, each linking to the meeting.
4. **Data explorer**. The raw → extracted → matched chain is kept in full and put on one screen:
   - Writer cards with dataset statistics.
   - A searchable table of all 140 meetings. Each row has a **View transcript** button that opens a right-hand drawer, so you can read a meeting without leaving the list. Compare rows open the same drawer at the job's turns.
   - Meeting view: the raw transcript, with job turns, decoys and extracted quotes marked, beside one card per real job. Each card lays out **answer key | words extracted | ID matched ✓/✗**, with the shortlist the matcher chose from.
   - Also in the meeting view: returned-but-not-real jobs, decoys, validation checks, and the exact prompt and input the extractor received.
   - Switch pipeline in place. Click a job to highlight its turns.
5. **Method**: question, hypotheses, unit of analysis, metrics, scoring rules, statistics, leakage controls, threats to validity, a **coverage matrix** (extractor × matcher, blank = not run) and a glossary.
6. **Appendix**: vocabulary, models and prices, one-off costs, prompts and schemas, export metadata, and a link to the old deck.

Every view can be linked, e.g. `#tab=explore&m=m012&p=llm@gpt-5.4|llm@gpt-5.4&f=m012-t2`.

## 4. Statistics changes

- **Intervals**: 95% percentile bootstrap over **meetings** (2,000 resamples, stratified by test set), not per-job binomial margins.
- **Paired comparisons**: every pipeline is resampled on the same meetings, so each gap (vs baseline, vs best, A vs B) gets its own interval.
- **Verdicts instead of ranks**: each pipeline is compared against both the best and the baseline. *Close to #1* means the interval of the gap to the best includes 0. That shows the data can't separate them, **not** that they are equal.
- **Recommendation rule**: the highest point estimate. Cost is never optimised automatically. At these prices (under $12 a year for the best pipeline), a "cheapest within N points" rule trades real accuracy for fractions of a cent. Cheaper options are presented as an explicit trade-off (the frontier table). Two earlier drafts were wrong: "cheapest that can't be proven worse" rewarded noisy pipelines, and "cheapest within 5 pts" picked a pipeline 4 pts lower to save about $10 a year.
- **Honest limit**: with 8 scored meetings, even a bootstrap is rough (it tends to be too narrow). Scoring the 20-meeting GPT-6 Astra set and pooling it is the single most valuable next step.

## 5. Recommended changes, by priority

1. **Quick wins** (current deck)
   - Move the verdict to slide 1.
   - Add Δ vs baseline.
   - Replace 1..N with group labels.
   - Generate all narrative numbers from data (fixes the 11-vs-2 LLM mismatch).
   - One vocabulary: job / equipment / matcher, with a glossary.
2. **Medium**
   - Field heatmap and outcome bars.
   - Merge the 7 per-meeting slides into one explorer view.
   - Put the full view state in the URL.
3. **Methodology**
   - Meeting-level paired bootstrap.
   - Explicit ship bar and hypotheses.
   - Coverage matrix.
   - Score and pool the second writer set.
4. **Structural**
   - Replace the slide deck with the tabbed report. Keep the deck only as a presenter view, if it's still wanted.
   - Fold `build.py` into `standup-exp walkthrough` once agreed.
