"""Stand-up extraction experiment.

Generates synthetic factory stand-up transcripts with a known answer key, then
compares extraction methods (LLM, GLiNER2, rules, fine-tuned GLiNER2) and
normalisation methods (alias, fuzzy, embeddings, LLM, GLiNER2, classifier).
"""

__version__ = "0.1.0"
