"""
Evaluation harness for sanctions name matching.

Computes precision-recall curves by:
  1. Vectorized scoring of all queries against the corpus using rapidfuzz's
     C++ backend (cdist). Full 1700 × 18899 score matrices are computed
     once per scorer and reduced to per-query best-match scores.
  2. Threshold sweep over pre-computed scores — O(n_thresholds) not
     O(n_thresholds × n_queries × corpus_size).

Precision / recall definitions used here:
    TP  — positive query where top-scoring entity is the correct SDN entry
          and score >= threshold
    FN  — positive query where top-scoring entity is wrong or score < threshold
    FP  — negative query where any entity scores >= threshold
    TN  — negative query where no entity scores >= threshold

    Precision  = TP / (TP + FP)
    Recall     = TP / (TP + FN)
    FPR        = FP / (FP + TN)   (false positive rate, x-axis of ROC)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import jellyfish
from rapidfuzz import distance as rfd, fuzz as rf_fuzz, process as rfp

from sanctions.matching.normalizers import normalize, tokenize
from sanctions.matching.scorers import ENSEMBLE_WEIGHTS


def _metaphone_str(name: str) -> str:
    """Space-joined Metaphone codes for each token."""
    codes = []
    for t in tokenize(name):
        try:
            code = jellyfish.metaphone(t)
            if code:
                codes.append(code)
        except Exception:
            pass
    return " ".join(codes) or name  # fallback to original if encoding fails


def build_corpus(akas: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (corpus_names, corpus_ids) from the primary Latin-script aliases.
    These are the names the screening engine searches against.
    """
    primary = akas[akas["is_primary"]].copy()
    return primary["aka_name"].tolist(), primary["entity_id"].tolist()


def score_all_queries(
    pos_queries: pd.DataFrame,
    neg_queries: pd.DataFrame,
    corpus_names: list[str],
    corpus_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score all queries against the corpus using vectorized cdist.

    Returns:
        pos_results — positive queries with columns:
            query_name, true_entity_id, variant_type, program,
            jw_match_id, jw_score, ts_match_id, ts_score,
            ph_match_id, ph_score, ens_match_id, ens_score
        neg_results — negative queries with columns:
            query_name, jw_score, ts_score, ph_score, ens_score
    """
    all_queries = pd.concat([
        pos_queries[["query_name"]],
        neg_queries[["query_name"]],
    ], ignore_index=True)

    q_names = all_queries["query_name"].tolist()
    n_q = len(q_names)
    n_c = len(corpus_names)

    # Pre-normalize everything once
    norm_q      = [normalize(q) for q in q_names]
    norm_corpus = [normalize(n) for n in corpus_names]
    ph_q        = [_metaphone_str(q) for q in norm_q]
    ph_corpus   = [_metaphone_str(n) for n in norm_corpus]

    print(f"  Scoring {n_q:,} queries × {n_c:,} corpus entries …")

    # Jaro-Winkler matrix  (normalized_similarity returns [0, 1])
    mat_jw = rfp.cdist(
        norm_q, norm_corpus,
        scorer=rfd.JaroWinkler.normalized_similarity,
        dtype=np.float32,
        workers=-1,
    )

    # Token-set matrix  (returns 0–100, normalise to 0–1)
    mat_ts = rfp.cdist(
        norm_q, norm_corpus, scorer=rf_fuzz.token_set_ratio,
        dtype=np.float32, workers=-1,
    ) / 100.0

    # Phonetic matrix
    mat_ph = rfp.cdist(
        ph_q, ph_corpus, scorer=rf_fuzz.token_set_ratio,
        dtype=np.float32, workers=-1,
    ) / 100.0

    # Ensemble
    w = ENSEMBLE_WEIGHTS
    mat_ens = (
        w["jaro_winkler"] * mat_jw
        + w["token_set"]  * mat_ts
        + w["phonetic"]   * mat_ph
    )

    # Reduce to best-match-per-query
    ids = np.array(corpus_ids)

    def _best(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = mat.argmax(axis=1)
        return ids[idx], mat[np.arange(n_q), idx]

    jw_ids, jw_scores   = _best(mat_jw)
    ts_ids, ts_scores   = _best(mat_ts)
    ph_ids, ph_scores   = _best(mat_ph)
    ens_ids, ens_scores = _best(mat_ens)

    n_pos = len(pos_queries)

    def _build_df(query_names, match_ids, scores, extra_df=None):
        df = pd.DataFrame({
            "query_name":    query_names,
            "jw_match_id":   jw_ids[:len(query_names)],
            "jw_score":      jw_scores[:len(query_names)].astype(float),
            "ts_match_id":   ts_ids[:len(query_names)],
            "ts_score":      ts_scores[:len(query_names)].astype(float),
            "ph_match_id":   ph_ids[:len(query_names)],
            "ph_score":      ph_scores[:len(query_names)].astype(float),
            "ens_match_id":  ens_ids[:len(query_names)],
            "ens_score":     ens_scores[:len(query_names)].astype(float),
        })
        if extra_df is not None:
            for col in extra_df.columns:
                if col != "query_name":
                    df[col] = extra_df[col].values
        return df

    pos_results = _build_df(
        q_names[:n_pos],
        jw_ids, jw_scores,
        pos_queries.reset_index(drop=True),
    )
    neg_results = _build_df(
        q_names[n_pos:],
        jw_ids[n_pos:], jw_scores[n_pos:],
    )

    return pos_results, neg_results


def sweep_thresholds(
    pos_results: pd.DataFrame,
    neg_results: pd.DataFrame,
    scorer: str = "ens",
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Sweep thresholds over pre-computed scores.

    scorer: one of 'jw', 'ts', 'ph', 'ens'
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.02, 0.02), 2)

    score_col = f"{scorer}_score"
    match_col = f"{scorer}_match_id"
    n_pos = len(pos_results)
    n_neg = len(neg_results)

    rows = []
    for t in thresholds:
        tp = int(
            ((pos_results[score_col] >= t) &
             (pos_results[match_col] == pos_results["true_entity_id"])).sum()
        )
        fn = n_pos - tp
        fp = int((neg_results[score_col] >= t).sum())
        tn = n_neg - fp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append({
            "threshold": t,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "scorer": scorer,
        })

    return pd.DataFrame(rows)


def find_operating_point(
    curve: pd.DataFrame,
    target_recall: float = 0.95,
) -> pd.Series:
    """Return the row with the highest threshold that still meets target_recall."""
    meeting = curve[curve["recall"] >= target_recall]
    if meeting.empty:
        return curve.iloc[0]
    return meeting.loc[meeting["threshold"].idxmax()]


def cost_at_operating_point(
    op: pd.Series,
    monthly_volume: int = 1_000_000,
    cost_per_alert: float = 35.0,
) -> dict:
    """
    Calculate the implied annual investigation cost at an operating point.

    monthly_volume: number of name-screening checks per month
    cost_per_alert: fully-loaded L1 investigation cost in USD
    """
    annual_alerts = op["fpr"] * monthly_volume * 12
    annual_cost   = annual_alerts * cost_per_alert
    return {
        "threshold":       op["threshold"],
        "recall":          op["recall"],
        "precision":       op["precision"],
        "fpr":             op["fpr"],
        "monthly_volume":  monthly_volume,
        "cost_per_alert":  cost_per_alert,
        "annual_fp_alerts": annual_alerts,
        "annual_cost_usd":  annual_cost,
    }
