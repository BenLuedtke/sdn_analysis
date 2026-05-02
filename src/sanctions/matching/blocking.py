"""
Candidate blocking strategies for sanctions name matching.

Blocking pre-filters the corpus before scoring to avoid O(n²) comparisons.
Every blocker trades recall for speed — a candidate that doesn't survive
blocking can never be returned, regardless of its true score.

Implemented strategies:
    FirstLetterBlocker  — Index by normalized first character.
                          Fast, ~1/26 corpus retained per query.
                          Recall impact: misses when first letter varies
                          (transliteration, 'al-' prefix, etc.)

    PhoneticBlocker     — Index by Metaphone code of first token.
                          Better recall on spelling variants than first-letter;
                          more selective than first-letter for names sharing
                          the same starting letter with different sounds.
"""

from __future__ import annotations

import jellyfish

from .normalizers import normalize, tokenize


class FirstLetterBlocker:
    """Block by first character of normalized name."""

    def __init__(self, corpus_names: list[str], corpus_ids: list[str]) -> None:
        self._ids = corpus_ids
        self._index: dict[str, list[int]] = {}
        for i, name in enumerate(corpus_names):
            norm = normalize(name)
            if norm:
                key = norm[0]
                self._index.setdefault(key, []).append(i)

    def candidate_indices(self, query: str) -> list[int]:
        norm = normalize(query)
        return self._index.get(norm[0], []) if norm else []

    def recall_estimate(self) -> float:
        """Fraction of corpus retained per average query."""
        total = sum(len(v) for v in self._index.values())
        n_buckets = len(self._index)
        return (total / n_buckets / total) if total > 0 else 0.0


class PhoneticBlocker:
    """Block by Metaphone code of the first normalized token."""

    def __init__(self, corpus_names: list[str], corpus_ids: list[str]) -> None:
        self._ids = corpus_ids
        self._index: dict[str, list[int]] = {}
        for i, name in enumerate(corpus_names):
            tokens = tokenize(name)
            if tokens:
                try:
                    code = jellyfish.metaphone(tokens[0])
                    if code:
                        self._index.setdefault(code, []).append(i)
                except Exception:
                    pass

    def candidate_indices(self, query: str) -> list[int]:
        tokens = tokenize(query)
        if not tokens:
            return []
        try:
            code = jellyfish.metaphone(tokens[0])
            return self._index.get(code, []) if code else []
        except Exception:
            return []
