# Ranking Determinism Report

Two independent pipeline runs on the same JD and candidate pool.
Metric: per-candidate rank shift between Run 1 and Run 2 (`Delta = Run2 - Run1`).

Model under test: `deepseek/deepseek-chat-v3-0324` | temperature=0.1

---

## deepseek/deepseek-chat-v3-0324

**52 candidates | 36 swaps | Mean |delta|: 2.4 | Max |delta|: 15 | Stable: 16/52**

| Candidate | Run 1 | Run 2 | Delta |
|---|---|---|---|
| this_vishalchauhan_2 | 10 | 25 | +15 |
| this_vishalchauhan_1 | 26 | 14 | -12 |
| michael_brown | 24 | 15 | -9 |
| carlos_mendez | 14 | 22 | +8 |
| test_agentic_stability | 20 | 12 | -8 |
| alex_williams | 35 | 41 | +6 |
| ryan_anderson | 19 | 13 | -6 |
| tom_nguyen | 30 | 36 | +6 |
| john_davis | 25 | 20 | -5 |
| ben_patel | 17 | 21 | +4 |
| hassan_ali | 41 | 37 | -4 |
| liu_yang | 22 | 26 | +4 |
| jennifer_lee | 13 | 16 | +3 |
| rahul_verma | 38 | 35 | -3 |
| arjun_patel | 7 | 9 | +2 |
| claire_thompson | 36 | 34 | -2 |
| jessica_moore | 40 | 38 | -2 |
| maria_garcia | 37 | 39 | +2 |
| raj_krishnan | 21 | 23 | +2 |
| sara_white | 16 | 18 | +2 |
| this_vishalchauhan_3 | 12 | 10 | -2 |
| this_vishalchauhan_4 | 9 | 7 | -2 |
| vishal_chauhan | 15 | 17 | +2 |
| aditya_sharma | 4 | 5 | +1 |
| anna_novak | 33 | 32 | -1 |
| chris_taylor | 28 | 27 | -1 |
| daniel_harris | 32 | 31 | -1 |
| david_wilson | 42 | 43 | +1 |
| fatima_malik | 34 | 33 | -1 |
| kevin_white | 18 | 19 | +1 |
| marcus_jackson | 43 | 42 | -1 |
| maya_jones | 39 | 40 | +1 |
| miguel_rodriguez | 27 | 28 | +1 |
| nina_petrov | 23 | 24 | +1 |
| sarah_kim | 5 | 4 | -1 |
| yuki_tanaka | 31 | 30 | -1 |
| aisha_ibrahim | 3 | 3 | 0 |
| alex_chen | 2 | 2 | 0 |
| candidate_1 | 52 | 52 | 0 |
| emily_johnson | 1 | 1 | 0 |
| gary_lewis | 51 | 51 | 0 |
| james_okafor | 29 | 29 | 0 |
| karen_brown | 50 | 50 | 0 |
| linda_wilson | 47 | 47 | 0 |
| nancy_garcia | 49 | 49 | 0 |
| patricia_hall | 44 | 44 | 0 |
| paul_robinson | 46 | 46 | 0 |
| peter_kozlov | 11 | 11 | 0 |
| priya_nair | 8 | 8 | 0 |
| robert_taylor | 48 | 48 | 0 |
| sophie_martin | 6 | 6 | 0 |
| steve_miller | 45 | 45 | 0 |

---

## deepseek/deepseek-chat-v3-0324 — Full Score Breakdown (Run 1)

Scores from Run 1.  
`match_score` = weighted LLM component score (0–100). `ce_score` = cross-encoder sigmoid (0–1). `vec_sim` = ChromaDB cosine similarity. `fused` = weighted RRF score ×1000 (llm 50% + ce 30% + vec 20%).

| Rank | Candidate | match_score | ce_score | vec_sim | fused |
|---|---|---|---|---|---|
| 1 | emily_johnson | 82.0 | 0.8059 | 0.5812 | 15.61 |
| 2 | alex_chen | 78.5 | 0.6731 | 0.5107 | 14.60 |
| 3 | aisha_ibrahim | 75.5 | 0.2051 | 0.6236 | 14.24 |
| 4 | aditya_sharma | 79.8 | 0.6643 | 0.4057 | 14.16 |
| 5 | sarah_kim | 87.5 | 0.0680 | 0.4130 | 13.88 |
| 6 | sophie_martin | 64.0 | 0.7919 | 0.5492 | 13.83 |
| 7 | arjun_patel | 82.2 | 0.0511 | 0.4988 | 13.47 |
| 8 | priya_nair | 86.8 | 0.0368 | 0.4001 | 13.36 |
| 9 | this_vishalchauhan_4 | 72.8 | 0.1624 | 0.4948 | 13.06 |
| 10 | this_vishalchauhan_2 | 85.8 | 0.0209 | 0.3947 | 12.98 |
| 11 | peter_kozlov | 62.0 | 0.1835 | 0.5220 | 12.81 |
| 12 | this_vishalchauhan_3 | 85.8 | 0.0209 | 0.3947 | 12.81 |
| 13 | jennifer_lee | 62.0 | 0.1702 | 0.5165 | 12.72 |
| 14 | carlos_mendez | 79.0 | 0.0472 | 0.4645 | 12.64 |
| 15 | vishal_chauhan | 85.8 | 0.0209 | 0.3947 | 12.64 |
| 16 | sara_white | 30.2 | 0.4132 | 0.5160 | 12.61 |
| 17 | ben_patel | 34.0 | 0.4996 | 0.4935 | 12.52 |
| 18 | kevin_white | 74.5 | 0.0695 | 0.4552 | 12.52 |
| 19 | ryan_anderson | 21.2 | 0.5190 | 0.5133 | 12.51 |
| 20 | test_agentic_stability | 81.0 | 0.0230 | 0.3991 | 12.50 |
| 21 | raj_krishnan | 67.8 | 0.0536 | 0.5386 | 12.41 |
| 22 | liu_yang | 69.5 | 0.1773 | 0.3738 | 12.39 |
| 23 | nina_petrov | 69.5 | 0.1505 | 0.4028 | 12.32 |
| 24 | michael_brown | 53.0 | 0.2227 | 0.5029 | 12.30 |
| 25 | john_davis | 23.5 | 0.3746 | 0.5062 | 12.26 |
| 26 | this_vishalchauhan_1 | 79.5 | 0.0209 | 0.3947 | 12.24 |
| 27 | miguel_rodriguez | 71.0 | 0.1485 | 0.3512 | 12.19 |
| 28 | chris_taylor | 21.2 | 0.4849 | 0.4768 | 11.98 |
| 29 | james_okafor | 75.2 | 0.0163 | 0.4370 | 11.89 |
| 30 | tom_nguyen | 64.8 | 0.0635 | 0.4901 | 11.82 |
| 31 | yuki_tanaka | 62.0 | 0.2625 | 0.3429 | 11.71 |
| 32 | daniel_harris | 28.8 | 0.0874 | 0.5045 | 11.41 |
| 33 | anna_novak | 55.0 | 0.0233 | 0.5045 | 11.40 |
| 34 | fatima_malik | 68.0 | 0.0203 | 0.4221 | 11.33 |
| 35 | alex_williams | 30.2 | 0.0596 | 0.5042 | 11.30 |
| 36 | claire_thompson | 28.8 | 0.0885 | 0.4794 | 11.26 |
| 37 | maria_garcia | 67.8 | 0.0178 | 0.4254 | 11.25 |
| 38 | rahul_verma | 68.8 | 0.0313 | 0.3453 | 11.25 |
| 39 | maya_jones | 31.0 | 0.0817 | 0.4497 | 11.19 |
| 40 | jessica_moore | 11.5 | 0.2969 | 0.3896 | 11.16 |
| 41 | hassan_ali | 67.0 | 0.0254 | 0.3790 | 11.01 |
| 42 | david_wilson | 55.0 | 0.0158 | 0.4690 | 10.69 |
| 43 | marcus_jackson | 19.0 | 0.0940 | 0.3925 | 10.60 |
| 44 | patricia_hall | 0.0 | 0.1661 | 0.2931 | 10.28 |
| 45 | steve_miller | 0.0 | 0.1457 | 0.2804 | 9.99 |
| 46 | paul_robinson | 0.0 | 0.0901 | 0.2756 | 9.92 |
| 47 | linda_wilson | 2.2 | 0.0612 | 0.3448 | 9.89 |
| 48 | robert_taylor | 11.2 | 0.0166 | 0.4098 | 9.86 |
| 49 | nancy_garcia | 0.0 | 0.0473 | 0.3573 | 9.74 |
| 50 | karen_brown | 7.5 | 0.0160 | 0.3492 | 9.44 |
| 51 | gary_lewis | 2.2 | 0.0122 | 0.3187 | 9.27 |
| 52 | candidate_1 | 0.0 | 0.0040 | 0.1117 | 9.09 |

---

## deepseek/deepseek-v3.2 — Reference (tested earlier session)

**52 candidates | 44 swaps | Mean |delta|: 3.4 | Max |delta|: 28 | Stable: 8/52**

| Candidate | Run 1 | Run 2 | Delta |
|---|---|---|---|
| test_agentic_stability | 17 | 45 | +28 |
| priya_nair | 3 | 24 | +21 |
| hassan_ali | 36 | 16 | -20 |
| nina_petrov | 22 | 6 | -16 |
| rahul_verma | 5 | 15 | +10 |
| john_davis | 26 | 34 | +8 |
| ben_patel | 44 | 37 | -7 |
| alex_williams | 42 | 36 | -6 |
| claire_thompson | 35 | 40 | +5 |
| jessica_moore | 37 | 33 | -4 |
| aditya_sharma | 6 | 3 | -3 |
| fatima_malik | 15 | 12 | -3 |
| this_vishalchauhan_1 | 10 | 7 | -3 |
| yuki_tanaka | 8 | 5 | -3 |
| (remaining 38 candidates: delta ≤ 2) | — | — | — |

---

## Summary

| Model | Candidates | Swaps | Mean delta | Max delta | Stable | Notes |
|---|---|---|---|---|---|---|
| deepseek/deepseek-v3.2 | 52 | 44 | 3.4 ranks | 28 ranks | 8/52 (15%) | Highly unstable |
| deepseek/deepseek-chat-v3-0324 | 52 | 36 | 2.4 ranks | 15 ranks | 16/52 (30%) | Real LLM scores; temperature=0.1 |

**Verdict**: `deepseek-chat-v3-0324` is meaningfully more stable than `deepseek-v3.2` (36 vs 44 swaps, max delta 15 vs 28).
Neither model is fully deterministic at temperature=0.1 — residual variance comes from the LLM evaluator (50% RRF weight).
CE + vector signals (50% combined) are fully deterministic across runs; they anchor the rankings and prevent worst-case swings.

> **Note on previous "0 swaps" result**: An earlier test showed 0 swaps for v3-0324, but this was an artifact — the model ID `deepseek/deepseek-v3-0324` (missing `chat-`) returned HTTP 400, so all LLM calls silently failed and match_score=0 for every candidate. With identical scores the tiebreak was alphabetical (fully deterministic but meaningless). The figures above reflect genuine LLM evaluation with the correct model ID `deepseek/deepseek-chat-v3-0324`.
