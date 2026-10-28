"""Fine-tune GLiNER2 on the train split (turn classification + entity spans).

Training examples are built from generated train meetings, one per transcript turn:

* classification ``turn``: "new job" for the turn that states a gold task
  (the first evidence turn containing the component words, else the first
  evidence turn), otherwise "not a job";
* entities (only on "new job" turns, only if the words appear verbatim):
  equipment, person, deadline, site, urgency.

The resulting checkpoint is used by the ``gliner_ft`` extractor.
Test meetings are never used, and train meetings never contain held-out aliases.
"""

from __future__ import annotations

import re
from pathlib import Path

from .dataset import load_all_train, load_meetings
from .extractors.gliner import ENTITY_LABELS, FT_TURN_LABELS
from .paths import MEETINGS_DIR, MODELS_DIR, env
from .schema import Meeting

JOB, NOT_JOB = list(FT_TURN_LABELS)


def _find(text: str, phrase: str | None) -> str | None:
    """Return the phrase exactly as it appears in the text (case preserved), if present."""
    if not phrase:
        return None
    m = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.I)
    return m.group(0) if m else None


def build_examples(meetings: list[Meeting], negatives_per_job: int = 3, seed: int = 7) -> list:
    """One example per job turn plus ``negatives_per_job`` sampled non-job turns per job.

    Only ~9% of turns are jobs; without downsampling the model learns "never a job".
    """
    import random

    from gliner2.training.data import Classification, InputExample

    rng = random.Random(seed)
    positives, negatives = [], []
    for meeting in meetings:
        job_turns: dict[int, dict] = {}
        for g in meeting.gold_tasks:
            if not g.evidence:
                continue
            main = next((i for i in g.evidence if g.component.text and _find(meeting.utterances[i].text, g.component.text)),
                        g.evidence[0])
            text = meeting.utterances[main].text
            ents = job_turns.setdefault(main, {})
            for label, phrase in (("equipment", g.component.text), ("person", g.owner.text), ("site", g.site.text),
                                  ("deadline", g.due_mention), ("urgency", g.priority_cue)):
                found = _find(text, phrase)
                if found:
                    ents.setdefault(label, []).append(found)
        for i, u in enumerate(meeting.utterances):
            is_job = i in job_turns
            ents = job_turns.get(i) or {}
            (positives if is_job else negatives).append(InputExample(
                text=u.text,
                entities=ents,
                entity_descriptions={k: ENTITY_LABELS[k] for k in ents},
                classifications=[Classification(task="turn", labels=[JOB, NOT_JOB], true_label=JOB if is_job else NOT_JOB)],
            ))
    negatives = rng.sample(negatives, min(len(negatives), negatives_per_job * len(positives)))
    examples = positives + negatives
    rng.shuffle(examples)
    return examples


def finetune(data_dir: Path | None = None, epochs: int = 1, batch_size: int = 8, log=print,
             out_name: str = "gliner-standup-all", negatives_per_job: int = 2, eval_steps: int = 90) -> Path:
    """Fine-tune GLiNER 2.5 base. ``data_dir=None`` = train meetings from every writer."""
    from gliner2 import AutoExtractor
    from gliner2.training.data import TrainingDataset
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    train = load_all_train() if data_dir is None else load_meetings(data_dir, "train")
    if not train:
        raise SystemExit("No train meetings. Run `standup-exp generate` first.")
    log(f"train meetings: {len(train)} ({', '.join(sorted({m.writer or 'template' for m in train}))})")
    examples = build_examples(train, negatives_per_job=negatives_per_job)
    dataset = TrainingDataset(examples)
    report = dataset.validate(raise_on_error=False)
    log(f"training examples: {len(examples)}  valid: {report['valid']}  invalid: {report['invalid']} "
        f"{report.get('errors', [])[:3]}")
    train_data, val_data, _ = dataset.split(train_ratio=0.9, val_ratio=0.1, test_ratio=0.0, shuffle=True, seed=7)

    out = MODELS_DIR / out_name
    model = AutoExtractor.from_pretrained(env("GLINER_MODEL", "fastino/gliner2.5-base-v1"))
    config = TrainingConfig(
        output_dir=str(out),
        experiment_name="standup",
        num_epochs=epochs,
        batch_size=batch_size,
        encoder_lr=1e-5,
        task_lr=5e-4,
        warmup_ratio=0.1,
        eval_strategy="steps" if eval_steps else "epoch",
        eval_steps=eval_steps or None,
        save_best=True,
        report_to_wandb=False,
        num_workers=0,  # Windows spawns workers and the tokenizer cannot be pickled
        pin_memory=False,
    )
    result = ExtractorTrainer(model, config).train(train_data=train_data, eval_data=val_data)
    log(f"done: {result}")
    return out / "best"
