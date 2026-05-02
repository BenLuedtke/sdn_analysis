"""Unit tests for name matching algorithms and the synthetic query generator."""

import pandas as pd
import pytest

from sanctions.matching import (
    ensemble_score,
    jaro_winkler_score,
    normalize,
    phonetic_score,
    score_all,
    token_set_score,
    tokenize,
)
from sanctions.eval import (
    generate_hard_negatives,
    generate_near_miss_negatives,
    generate_positive_queries,
)


# ── Normalizer ────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_lowercase(self):
        assert normalize("JOHN SMITH") == "john smith"

    def test_strips_diacritics(self):
        assert normalize("José García") == "jose garcia"

    def test_removes_punctuation(self):
        assert normalize("AL-RASHID, Omar") == "al rashid omar"

    def test_collapses_whitespace(self):
        assert normalize("  JOHN   SMITH  ") == "john smith"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_hyphenated_name(self):
        result = normalize("AL-HUSSEIN")
        assert "hussein" in result


# ── Scorers ───────────────────────────────────────────────────────────────────

class TestJaroWinkler:
    def test_exact_match(self):
        assert jaro_winkler_score("JOHN SMITH", "JOHN SMITH") == pytest.approx(1.0)

    def test_case_invariant(self):
        assert jaro_winkler_score("john smith", "JOHN SMITH") == pytest.approx(1.0)

    def test_high_score_for_close_names(self):
        # One character different
        assert jaro_winkler_score("JOHN SMITH", "JON SMITH") > 0.85

    def test_low_score_for_different_names(self):
        assert jaro_winkler_score("JOHN SMITH", "AHMED HASSAN") < 0.6

    def test_empty_returns_zero(self):
        assert jaro_winkler_score("", "JOHN SMITH") == 0.0
        assert jaro_winkler_score("JOHN SMITH", "") == 0.0


class TestTokenSet:
    def test_exact_match(self):
        assert token_set_score("JOHN SMITH", "JOHN SMITH") == pytest.approx(1.0)

    def test_word_reordering(self):
        # Token set ratio is invariant to word order
        assert token_set_score("SMITH JOHN", "JOHN SMITH") == pytest.approx(1.0)

    def test_subset_match(self):
        # "JOHN SMITH" is a subset of "JOHN MICHAEL SMITH"
        assert token_set_score("JOHN SMITH", "JOHN MICHAEL SMITH") > 0.85

    def test_different_names(self):
        assert token_set_score("JOHN SMITH", "AHMED HASSAN") < 0.4


class TestPhonetic:
    def test_exact_match(self):
        assert phonetic_score("JOHN SMITH", "JOHN SMITH") == pytest.approx(1.0)

    def test_phonetic_variants(self):
        # Hassan / Hasan — same phonetic code
        assert phonetic_score("HASSAN", "HASAN") > 0.8

    def test_transliteration_variant(self):
        # Mohamed / Mohammed — phonetically similar
        assert phonetic_score("MOHAMMED", "MOHAMED") > 0.7

    def test_different_names(self):
        # Phonetic codes for JOHN SMITH vs HASSAN OMAR share no tokens —
        # score should be below 0.5 (not as tight as character-level matchers)
        assert phonetic_score("JOHN SMITH", "HASSAN OMAR") < 0.5

    def test_empty_returns_zero(self):
        assert phonetic_score("", "JOHN") == 0.0


class TestEnsemble:
    def test_exact_match(self):
        assert ensemble_score("JOHN SMITH", "JOHN SMITH") == pytest.approx(1.0)

    def test_score_between_zero_and_one(self):
        score = ensemble_score("JOHN SMITH", "JON SMYTH")
        assert 0.0 <= score <= 1.0

    def test_score_all_returns_four_keys(self):
        result = score_all("JOHN SMITH", "JON SMYTH")
        assert set(result.keys()) == {"jaro_winkler", "token_set", "phonetic", "ensemble"}

    def test_custom_weights(self):
        # Forcing all weight onto jaro_winkler
        score_jw_only = ensemble_score(
            "JOHN SMITH", "SMITH JOHN",
            weights={"jaro_winkler": 1.0, "token_set": 0.0, "phonetic": 0.0},
        )
        score_ts_only = ensemble_score(
            "JOHN SMITH", "SMITH JOHN",
            weights={"jaro_winkler": 0.0, "token_set": 1.0, "phonetic": 0.0},
        )
        # Token set should score higher on reordering than jaro_winkler
        assert score_ts_only > score_jw_only


# ── Query generator ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_akas():
    """Minimal akas DataFrame for testing."""
    return pd.DataFrame([
        {"entity_id": "1", "aka_name": "JOHN MICHAEL SMITH", "is_primary": True,
         "is_weak": False, "script": "Latin", "alias_type": "Name"},
        {"entity_id": "2", "aka_name": "AHMED HASSAN OMAR", "is_primary": True,
         "is_weak": False, "script": "Latin", "alias_type": "Name"},
        {"entity_id": "3", "aka_name": "IVAN PETROV", "is_primary": True,
         "is_weak": False, "script": "Latin", "alias_type": "Name"},
    ] * 40)  # repeat to give stratified sampler enough rows


@pytest.fixture(scope="module")
def sample_entities():
    return pd.DataFrame([
        {"entity_id": "1", "entity_type": "Individual", "programs": ["SDGT"]},
        {"entity_id": "2", "entity_type": "Individual", "programs": ["IRAN"]},
        {"entity_id": "3", "entity_type": "Individual", "programs": ["RUSSIA-EO14024"]},
    ] * 40)


class TestQueryGen:
    def test_positive_row_count(self, sample_akas, sample_entities):
        df = generate_positive_queries(
            sample_akas, sample_entities, n_entities=10, variants_per_entity=5, seed=42
        )
        assert len(df) == 50

    def test_positive_has_required_columns(self, sample_akas, sample_entities):
        df = generate_positive_queries(
            sample_akas, sample_entities, n_entities=10, variants_per_entity=5, seed=42
        )
        assert {"query_name", "true_entity_id", "true_name", "variant_type"}.issubset(df.columns)

    def test_positive_is_deterministic(self, sample_akas, sample_entities):
        df1 = generate_positive_queries(
            sample_akas, sample_entities, n_entities=10, variants_per_entity=5, seed=42
        )
        df2 = generate_positive_queries(
            sample_akas, sample_entities, n_entities=10, variants_per_entity=5, seed=42
        )
        assert df1["query_name"].tolist() == df2["query_name"].tolist()

    def test_hard_negatives_count(self):
        df = generate_hard_negatives(n=100, seed=42)
        assert len(df) == 100

    def test_hard_negatives_deterministic(self):
        df1 = generate_hard_negatives(n=50, seed=99)
        df2 = generate_hard_negatives(n=50, seed=99)
        assert df1["query_name"].tolist() == df2["query_name"].tolist()

    def test_near_miss_count(self):
        df = generate_near_miss_negatives(n=40, seed=42)
        assert len(df) == 40

    def test_near_miss_has_both_types(self):
        df = generate_near_miss_negatives(n=100, seed=42)
        assert "arabic" in df["near_miss_type"].values
        assert "russian" in df["near_miss_type"].values
