"""
Pure IR evaluation metrics — no I/O, no side effects.

All rank-cutoff metric functions take:
  ranked_ids  : list[str]      — candidate IDs in ranked order (index 0 = rank 1)
                                 MUST be pre-filtered to only labeled (qrels) candidates
  qrels       : dict[str, int] — {candidate_id: relevance_grade}  grades 0-4
  k           : int            — rank cutoff

Candidates absent from ranked_ids are treated as grade 0 (not retrieved).
Candidates absent from qrels are treated as unjudged — caller must filter them out.
"""
import math
from typing import Callable

import numpy as np
from scipy import stats


# ── DCG / NDCG ────────────────────────────────────────────────────────────────

def _dcg(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ideal_dcg_at_k(qrels: dict[str, int], k: int) -> float:
    sorted_grades = sorted(qrels.values(), reverse=True)
    return _dcg(sorted_grades, k)


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k.
    Primary metric — handles graded relevance across 4 tiers."""
    idcg = ideal_dcg_at_k(qrels, k)
    if idcg == 0:
        return 0.0
    gains = [qrels.get(cid, 0) for cid in ranked_ids]
    return min(1.0, _dcg(gains, k) / idcg)


# ── Recall ────────────────────────────────────────────────────────────────────

def recall_at_k(
    ranked_ids: list[str],
    qrels: dict[str, int],
    k: int,
    relevant_grade: int = 4,
) -> float:
    """Fraction of tier-1 (grade >= relevant_grade) candidates found in top-k.
    Denominator is min(k, total_relevant) so perfect recall is always achievable."""
    relevant = {cid for cid, g in qrels.items() if g >= relevant_grade}
    if not relevant:
        return 0.0
    found = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return found / min(k, len(relevant))


# ── Average Precision ─────────────────────────────────────────────────────────

def ap_at_k(
    ranked_ids: list[str],
    qrels: dict[str, int],
    k: int,
    relevant_grade: int = 3,
) -> float:
    """Average Precision at k (binary relevance at given grade threshold)."""
    relevant = {cid for cid, g in qrels.items() if g >= relevant_grade}
    if not relevant:
        return 0.0
    hits, total_prec = 0, 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        if cid in relevant:
            hits += 1
            total_prec += hits / (i + 1)
    return total_prec / min(k, len(relevant))


# ── Mean Reciprocal Rank ──────────────────────────────────────────────────────

def mrr(
    ranked_ids: list[str],
    qrels: dict[str, int],
    relevant_grade: int = 4,
) -> float:
    """Reciprocal rank of the first hit at grade >= relevant_grade.
    Uses tier-1 (grade=4) by default — 'where is the first top candidate?'"""
    relevant = {cid for cid, g in qrels.items() if g >= relevant_grade}
    for i, cid in enumerate(ranked_ids):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Rank Correlation ──────────────────────────────────────────────────────────

def _build_rank_grade_pairs(ranked_ids: list[str], qrels: dict[str, int]) -> tuple[list, list]:
    """Align ranks and grades for only the candidates present in both lists."""
    pairs = [(i + 1, qrels[cid]) for i, cid in enumerate(ranked_ids) if cid in qrels]
    if not pairs:
        return [], []
    ranks, grades = zip(*pairs)
    return list(ranks), list(grades)


def kendall_tau(ranked_ids: list[str], qrels: dict[str, int]) -> float:
    """Kendall's tau-b between pipeline rank and relevance grade.
    Positive = pipeline ranks high-relevance candidates earlier (lower rank number)."""
    ranks, grades = _build_rank_grade_pairs(ranked_ids, qrels)
    if len(ranks) < 2:
        return 0.0
    tau, _ = stats.kendalltau([-r for r in ranks], grades)
    return round(float(tau), 4) if not math.isnan(tau) else 0.0


def spearman_r(ranked_ids: list[str], qrels: dict[str, int]) -> float:
    """Spearman's rho between pipeline rank and relevance grade."""
    ranks, grades = _build_rank_grade_pairs(ranked_ids, qrels)
    if len(ranks) < 2:
        return 0.0
    rho, _ = stats.spearmanr([-r for r in ranks], grades)
    return round(float(rho), 4) if not math.isnan(rho) else 0.0


# ── Jackknife Confidence Interval ─────────────────────────────────────────────

def jackknife_ci(
    metric_fn: Callable,
    ranked_ids: list[str],
    qrels: dict[str, int],
    k: int,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Leave-one-out jackknife 95% CI for a single-query IR metric.

    Removes each candidate from both the ranked list and qrels, recomputes
    the metric, estimates standard error, and returns a Gaussian CI.
    Returns (point_estimate, lower_bound, upper_bound).

    Standard approach for single-query evaluation where there are no multiple
    query replicates to bootstrap over (Cormack & Lynam, SIGIR 2006 style).
    """
    all_cids = list(qrels.keys())
    n = len(all_cids)
    point = metric_fn(ranked_ids, qrels, k)

    if n < 2:
        return round(point, 4), round(point, 4), round(point, 4)

    jack_scores = np.empty(n)
    for i, cid in enumerate(all_cids):
        loo_ranked = [c for c in ranked_ids if c != cid]
        loo_qrels = {c: g for c, g in qrels.items() if c != cid}
        jack_scores[i] = metric_fn(loo_ranked, loo_qrels, k)

    jack_mean = jack_scores.mean()
    se = math.sqrt((n - 1) / n * np.sum((jack_scores - jack_mean) ** 2))

    z = stats.norm.ppf(1 - (1 - ci) / 2)
    lower = max(0.0, point - z * se)
    upper = min(1.0, point + z * se)
    return round(point, 4), round(lower, 4), round(upper, 4)


def jackknife_ci_corr(
    corr_fn: Callable,
    ranked_ids: list[str],
    qrels: dict[str, int],
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Jackknife CI for correlation metrics (kendall_tau, spearman_r).
    These don't take a k parameter."""
    all_cids = list(qrels.keys())
    n = len(all_cids)
    point = corr_fn(ranked_ids, qrels)

    if n < 2:
        return round(point, 4), round(point, 4), round(point, 4)

    jack_scores = np.empty(n)
    for i, cid in enumerate(all_cids):
        loo_ranked = [c for c in ranked_ids if c != cid]
        loo_qrels = {c: g for c, g in qrels.items() if c != cid}
        jack_scores[i] = corr_fn(loo_ranked, loo_qrels)

    jack_mean = jack_scores.mean()
    se = math.sqrt((n - 1) / n * np.sum((jack_scores - jack_mean) ** 2))

    z = stats.norm.ppf(1 - (1 - ci) / 2)
    lower = max(-1.0, point - z * se)
    upper = min(1.0, point + z * se)
    return round(point, 4), round(lower, 4), round(upper, 4)


# ── Permutation Test ──────────────────────────────────────────────────────────

def permutation_test(
    metric_fn: Callable,
    sys_a_ranked: list[str],
    sys_b_ranked: list[str],
    qrels: dict[str, int],
    k: int,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """One-sided permutation test: P(random_delta >= observed_delta).

    Tests whether sys_a significantly outperforms sys_b on metric_fn@k.
    Under the null hypothesis, candidate-to-system assignments are random.
    Candidates are randomly reassigned between sys_a and sys_b positions.
    """
    rng = np.random.default_rng(seed)
    score_a = metric_fn(sys_a_ranked, qrels, k)
    score_b = metric_fn(sys_b_ranked, qrels, k)
    observed_delta = score_a - score_b

    # Combined pool for permutation
    pool = list(qrels.keys())
    n = len(pool)
    count_gte = 0
    for _ in range(n_perm):
        perm = rng.permutation(pool).tolist()
        perm_a = perm[:min(len(sys_a_ranked), n)]
        perm_b = perm[:min(len(sys_b_ranked), n)]
        delta = metric_fn(perm_a, qrels, k) - metric_fn(perm_b, qrels, k)
        if delta >= observed_delta:
            count_gte += 1

    return count_gte / n_perm
