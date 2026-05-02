"""
Baseline name-matching scorers for sanctions screening.

Three independent algorithms, each returning a float in [0, 1]:

    jaro_winkler  — Character-level similarity with prefix bonus.
                    Best for short strings where prefix matters (surnames).

    token_set     — Bag-of-words overlap after sorting tokens.
                    Handles word reordering: "SMITH JOHN" == "JOHN SMITH".

    phonetic      — Metaphone-encoded token comparison via Jaro-Winkler.
                    Catches transliteration variants in Latin script.
                    Known limitation: Metaphone was designed for English;
                    accuracy degrades on Arabic-origin names — see notebook 3.

Ensemble weights (0.4 / 0.4 / 0.2) were set by hand based on operational
reasoning: character-level and token-level signals are equally important for
sanctions name matching; phonetic adds marginal lift and is weighted lower
because it degrades on non-English names. SR 26-2 requires documented,
defensible weights; these are not learned from data.
"""

from __future__ import annotations

import jellyfish
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz.distance import JaroWinkler

from .normalizers import normalize, tokenize

ENSEMBLE_WEIGHTS: dict[str, float] = {
    "jaro_winkler": 0.4,
    "token_set": 0.4,
    "phonetic": 0.2,
}


def jaro_winkler_score(a: str, b: str) -> float:
    """Jaro-Winkler similarity on normalized names."""
    a_n, b_n = normalize(a), normalize(b)
    if not a_n or not b_n:
        return 0.0
    return JaroWinkler.normalized_similarity(a_n, b_n)


def token_set_score(a: str, b: str) -> float:
    """Token set ratio on normalized names. Invariant to word order."""
    a_n, b_n = normalize(a), normalize(b)
    if not a_n or not b_n:
        return 0.0
    return rf_fuzz.token_set_ratio(a_n, b_n) / 100.0


def _metaphone_string(name: str) -> str:
    """Return space-joined Metaphone codes for each token in name."""
    codes = []
    for token in tokenize(name):
        try:
            code = jellyfish.metaphone(token)
            if code:
                codes.append(code)
        except Exception:
            pass
    return " ".join(codes)


def phonetic_score(a: str, b: str) -> float:
    """
    Phonetic similarity via Metaphone encoding.
    Encodes each token independently, then compares the code strings
    with token_set_ratio to handle reordering.
    """
    a_codes = _metaphone_string(a)
    b_codes = _metaphone_string(b)
    if not a_codes or not b_codes:
        return 0.0
    return rf_fuzz.token_set_ratio(a_codes, b_codes) / 100.0


def ensemble_score(
    a: str,
    b: str,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Weighted ensemble of all three matchers.

    Weights must sum to 1.0. Defaults to ENSEMBLE_WEIGHTS.
    All three component scores are also returned as a dict if needed —
    call the individual functions directly for that.
    """
    w = weights or ENSEMBLE_WEIGHTS
    return (
        w["jaro_winkler"] * jaro_winkler_score(a, b)
        + w["token_set"] * token_set_score(a, b)
        + w["phonetic"] * phonetic_score(a, b)
    )


def score_all(a: str, b: str) -> dict[str, float]:
    """Return all three component scores plus the ensemble score."""
    jw = jaro_winkler_score(a, b)
    ts = token_set_score(a, b)
    ph = phonetic_score(a, b)
    ens = (
        ENSEMBLE_WEIGHTS["jaro_winkler"] * jw
        + ENSEMBLE_WEIGHTS["token_set"] * ts
        + ENSEMBLE_WEIGHTS["phonetic"] * ph
    )
    return {"jaro_winkler": jw, "token_set": ts, "phonetic": ph, "ensemble": ens}
