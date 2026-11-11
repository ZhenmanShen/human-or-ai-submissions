# Audit: which decisions behind the six submitted models used the external proxy

Written 2026-11-10, before the proxy was removed from the pipeline.

**What "the proxy" is.** A labelled evaluation set we built ourselves (`data/eval/`, never in git):
- AI text from 7 local open-weight generators that are not in `train.csv`
  (`pretrained/scripts/build_eval_set.py`, `generate_ai_text.py`);
- human poems and stories from held-out Project Gutenberg books (`fetch_gutenberg.py`).

It was scored by `pretrained/scripts/eval_ood.py` (pooled `proxy_auc`, `pretrained/experiments/ood_eval.csv`) and
`classical/ood_eval.py` (`classical/results/ood_eval.csv`). It was never trained on. It was used to choose things.
That is the problem under competition rule 4(b): "No additional externally sourced labeled datasets may be used to
train, validate, or determine competition predictions."

Timeline (commit dates):
- `1fc181f` 2026-10-29: full PC benchmark committed (session `20261028-145019_pc`, 35 jobs, 5-fold stratified CV).
  At this point no eval set existed.
- `ed4ad2d` 2026-10-30: first commit of the eval set and of OOD scoring.
- `a7f84f3` 2026-10-31: `eval_ood.py` gets `proxy_auc` and proxy-driven greedy blending.
- `77663a4` 2026-10-31: blends A, B and C built; commit title "final submissions chosen on a Kaggle proxy".
- `6724fe5` / `08cde91` 2026-11-01: C2 `stylo_gbm` added and its Kaggle file written.
- `1bc0f61` / `403cfb5` 2026-11-01: C1 v1 and C3 v1 added.
- `fb3df12` / `efb72f0` 2026-11-09: classical `ood_eval.py`, and the design variants of C1 and C3 compared on it.
- `6316699` / `dc302d2` 2026-11-09: C1 v2 and C3 v2 written.

Every pretrained member config (`pretrained/configs/models/*.yaml`) was last changed on 2026-10-28 (`7fd6185`),
before the proxy existed. The one exception is `lm_gpt2_nostyle.yaml` (`fe19235`, 2026-11-01), added because
blend A's proxy-greedy selection picked it. The member runs used by the blends (sessions `20261030-135404_ood`,
`20261031-144451_ood`, `20261031_var`) also scored the eval set. Scoring it does not change training: their
5-fold CV AUROCs equal the pc-session runs of the same configs (`pretrained/experiments/results.csv`).

## Summary

| Model | Proxy used for a decision? | Decisions that used it |
|---|---|---|
| Blend A | **Yes** | members and weights (greedy forward selection on `proxy_auc`) |
| Blend B | **No** (see caveat) | none: members = top 4 runs by 5-fold CV in the pre-proxy pc session, equal weights |
| Blend C | **Yes, in effect** | the 5th member, `ft_roberta_base`, is not the CV choice; it matches the proxy ranking |
| C1 `tfidf_lr` v2 | **Yes** | feature views (raw char n-grams, distorted-text view, `distort_k`=300, `distort_ngrams`=3) |
| C2 `stylo_gbm` | **No** | none |
| C3 `transformer_scratch` v2 | **Yes** | input design (`join_lines`, `distort_k`=300) chosen among 11 variants |

## Per model

### Blend A (`2026-10-31_final_A_greedy_proxy.csv`)

| Decision | Proxy? | Evidence |
|---|---|---|
| Members and weights 3 3 2 1 1 1 1 | yes | greedy forward selection with replacement on `proxy_auc` (`eval_ood.py`, `a7f84f3`); row `blend:greedy_proxy` in `pretrained/experiments/ood_eval.csv`; file name "greedy_proxy" |
| Including `lm_gpt2_nostyle` | yes | its config was added only as "LM-features-only member of final_A" (`fe19235`) |
| Member hyperparameters | no | configs dated 2026-10-28 (`7fd6185`, `286943b`) |
| CDF blending (`cdf_mean`) | no score comparison | introduced in `77663a4` "as eval_ood does"; the CDF uses only train OOF scores, and `ood_eval.csv` has no rank-vs-CDF comparison |

Verdict: A depends on the proxy. It stays in the log as history and is not a final selection.

### Blend B (`2026-10-31_final_B_lm4.csv`)

| Decision | Proxy? | Evidence |
|---|---|---|
| Members `lm_gpt2`, `lm_gpt2_medium`, `binoculars_qwen05b`, `binoculars_qwen15b` | no | these are the 4 highest 5-fold CV AUROCs of **all** 36 runs in session `20261028-145019_pc` (any family): 0.99935, 0.99924, 0.99924, 0.99920. Next is `ft_distilroberta` at 0.99911 (`pretrained/experiments/results.csv`, committed in `1fc181f` before the eval set existed) |
| Equal weights | no | no weights were fitted |
| Member hyperparameters | no | configs dated 2026-10-28 |
| CDF blending | no score comparison | as for A |

**Caveat, stated plainly.** B was assembled on 2026-10-31, after the proxy scores were known, and the same four
models were also the top four on the proxy (commit message of `77663a4`). The rule "top 4 by CV in the pc
session" reproduces B exactly and uses no proxy. But the record does not show that this rule was written down
before the proxy results were seen. The cut at four is also not a clear gap: #4 and #5 differ by 0.00009 of CV
AUROC, below the fold s.d. (about 0.0005). What can be said honestly: without the proxy, B's members and
weights would be the same.

Verdict: B does not depend on the proxy. It is the final pretrained selection.

### Blend C (`2026-10-31_final_C_lm4_roberta.csv`)

| Decision | Proxy? | Evidence |
|---|---|---|
| B's four members | no | as above |
| 5th member `ft_roberta_base` | **yes, in effect** | by 5-fold CV, `ft_distilroberta` (pc 0.99911, ood session 0.99880) and `ft_modernbert_base` (pc 0.99899) beat `ft_roberta_base` (pc 0.99875, ood session 0.99851). On the proxy `ft_roberta_base` was best of the three (0.97354 vs 0.97094 / 0.97081, `pretrained/experiments/ood_eval.csv`), and it was in A's proxy-greedy selection. No proxy-free reason for picking it is recorded. The stated reason, "adds a second family", is true of all three |
| Equal weights | no | |

Verdict: C's choice of `ft_roberta_base` over the CV-better fine-tunes is explained only by the proxy. It is
reported as is. C is not a final selection, and it is not retrained.

### C1 `tfidf_lr` v2 (`classical/kaggle_submissions/final_E_tfidf_lr.csv`)

| Decision | Proxy? | Evidence |
|---|---|---|
| v1 design (word 1-2 + `char_wb` 2-5) | no | `1bc0f61`, 2026-11-01 |
| v1 C=64, lowercase=False | no | 5-fold stratified inner CV on the 8,000 rows |
| Redesign at all | **Kaggle feedback** | v2 was started because v1 scored public 0.88583 against val 0.96256 (commit message of `fb3df12`). The redesign was prompted by a leaderboard score, but no setting was chosen on the leaderboard |
| v2 views: raw `char` 1-5 instead of `char_wb` 2-5, + distorted word 1-3-grams | **yes** | compared only in `classical/results/ood_eval.csv` (`efb72f0`); none of these variants has a `results/tfidf_lr@*` validation run |
| `distort_k` = 300 | **yes, indirectly** | only k=300 was tried for C1; 300 was the value the C3 variants (k=100/300/1000) favoured on the proxy |
| `distort_ngrams` = 3 | **yes** | `tfidf_lr@char_d300` (proxy 0.95495, `distort_ngrams` not recorded) vs `tfidf_lr` (0.95545) |
| v2 C=64, lowercase=True | no | 5-fold stratified inner CV (`classical/results/tfidf_lr/search.csv`, `dc302d2`) |

Settings compared only in `classical/results/ood_eval.csv`:
- `tfidf_lr@v1`: `char_wb` 2-5, no distortion, C=64, lowercase=False;
- `tfidf_lr@char_d300`: raw char 1-5 + distorted (k=300), C=64, lowercase=False;
- `tfidf_lr@char_d300_c4`: the same with C=4, lowercase=True;
- `tfidf_lr`: raw char 1-5 + distorted (k=300, 1-3-grams), C=64, lowercase=True (= v2).

Verdict: C1 v2's features were chosen on the proxy. It is re-selected without the proxy in Step 3.

### C2 `stylo_gbm` (`classical/kaggle_submissions/final_D_stylo_gbm.csv`)

| Decision | Proxy? | Evidence |
|---|---|---|
| 101 features | no | `6724fe5` (2026-11-01); `features/stylometry.py` unchanged since (apart from the folder rename) |
| leaves / learning rate / trees | no | 5-fold stratified inner CV (`classical/results/stylo_gbm/search.csv`) |
| Kaggle file | no | `08cde91` 2026-11-01, eight days before the classical `ood_eval.py` existed (`fb3df12`) |

The `stylo_gbm` row in `classical/results/ood_eval.csv` was computed after the file was written (`efb72f0`),
for reporting only. The ablations `stylo_gbm@no_layout` and `@no_added` were scored on the fixed validation rows,
for reporting only, and chose nothing.

Verdict: C2 does not depend on the proxy. No rerun.

### C3 `transformer_scratch` v2 (`classical/kaggle_submissions/final_F_transformer_scratch.csv`)

| Decision | Proxy? | Evidence |
|---|---|---|
| v1 architecture and training (4 layers, d=256, lr 3e-4, ...) | no | fixed up front in `403cfb5`, no search |
| Redesign at all | **Kaggle feedback** | v1 public 0.83201 against val 0.98500 (`fb3df12`) |
| `join_lines`=True, `distort_k`=300 | **yes** | the best of 11 variants by `proxy_auc` in `classical/results/ood_eval.csv` (`efb72f0`) |
| Seeds `d300_join_s1`, `_s2` | yes, for reporting | proxy spread of the chosen design |

Variants compared only in `classical/results/ood_eval.csv`: `v1`, `d300`, `c128`, `d300_c128`,
`d300_c128_small`, `join`, `d300_join` (= v2), `d100_join`, `d1000_join`, `d300_join_c256`, `d300_join_do2`.

Verdict: C3 v2's input design was chosen on the proxy. It is re-selected without the proxy in Step 3.

## Not submitted, for completeness

`ensemble_c123` (classical blend, never uploaded) has weights fitted by greedy selection on out-of-fold
predictions inside the 8,000 rows, with no proxy. Its members are C1 v2 and C3 v2, so it inherits their
dependence. It is not a final selection.
