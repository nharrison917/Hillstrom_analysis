# RERUN.md — Feature Engineering Overhaul

## What Changed and Why

### Problem addressed
The original pipeline had two structural feature representation problems:

1. **zip_code and channel were label-encoded** — assigned arbitrary integers
   (Urban=0, Suburban=1, Rural=2 or similar depending on alphabetical order).
   Tree-based models treat these as ordinal: "Suburban is between Urban and Rural."
   That ordering is false and masks real signal. The Suburban/Urban flip in email
   type responsiveness requires two splits to express with ordinal encoding, and
   the model has no reason to prioritize finding them.

2. **No recency x newbie interaction term** — Stage 6b (RCT cross-tabs) established
   that the recency x newbie interaction is the strongest moderator in the dataset.
   Lapsed newbies (10-12mo) are the highest Mens responders (+$1.26 lift). Recent
   established customers have *negative* Womens lift (-$0.21). The causal forest
   had both features marginally but no multiplicative term to detect this directly.

Both problems affect the causal forest (Stage 3), which means every downstream
stage inherits the misspecification: Stage 4 EV scores, Stage 5 clustering,
Stage 6 CATE clustering, and both policy tree stages.

history_segment was intentionally kept as ordinal label-encoding — the spend
tier labels have a genuine ordering that one-hot encoding would discard.

### Design choices for reference categories
- **Phone** as channel reference (dropped in one-hot): weakest email responder.
  Phone Mens lift +0.55pp vs Multichannel +1.02pp. Phone Womens lift nearly zero.
- **Rural** as zip reference (dropped in one-hot): lowest conversion lift;
  catalog-channel affinity explains its high base rate, not email responsiveness.

These choices don't affect model performance (trees are permutation-invariant to
reference category selection) but make policy rule output more readable: splits
will name channel_Multichannel and zip_Urban explicitly.

---

## What Was Built

### preprocessing.py (new file)
Single shared module imported by all model scripts. Contains:
- `DATA_FILE` — dataset path constant
- `RANDOM_STATE = 42`
- `FEATURES` — the canonical 12-feature list (was 9)
- `prepare_features(df)` — all derived column logic in one place

All scripts that previously repeated 6-8 lines of inline feature engineering
now call `prepare_features()` instead. One edit to the feature set propagates
everywhere.

### Updated feature set (12 features, was 9)

| Feature | Type | Change |
|---------|------|--------|
| recency | continuous | unchanged |
| log_history | continuous | unchanged |
| mens | binary | unchanged |
| womens | binary | unchanged |
| both_catalogs | binary | unchanged |
| newbie | binary | unchanged |
| history_segment_enc | ordinal int | unchanged (genuine ordering) |
| zip_Suburban | binary (one-hot) | NEW — replaces zip_code_enc |
| zip_Urban | binary (one-hot) | NEW — replaces zip_code_enc |
| channel_Multichannel | binary (one-hot) | NEW — replaces channel_enc |
| channel_Web | binary (one-hot) | NEW — replaces channel_enc |
| recency_x_newbie | continuous | NEW — recency * newbie |

### Scripts updated
All scripts import `from preprocessing import prepare_features, FEATURES, RANDOM_STATE, DATA_FILE`
and call `df = prepare_features(pd.read_csv(DATA_FILE))`.

- `01_baseline.py` — imports updated
- `02_conversion_uplift.py` — imports updated
- `03_revenue_uplift.py` — imports updated
- `04_expected_value.py` — imports updated
- `05_feature_clustering.py` — imports updated; CLUSTER_FEATURES = FEATURES
- `07_policy_tree.py` — imports updated; tree_to_rules rewritten for one-hot columns
- `07b_policy_tree_tuning.py` — imports updated; tree_to_rules rewritten for one-hot columns

Not changed: `06_cate_clustering.py` (clusters on CATE scores, not features),
`06b_recency_newbie.py` (RCT cross-tabs, model-free).

### tree_to_rules rewrite (Stages 7 and 7b)
The old implementation used index-keyed reverse lookup dicts for zip and channel
(because label-encoded ints needed translation back to strings). With one-hot
encoding the column names are already the display strings. New implementation:
- Binary one-hot columns: `channel_Multichannel = 0` / `channel_Multichannel = 1`
- log_history: still back-transformed to dollars
- history_segment_enc: still uses rev_hist reverse lookup
- All other numerics: `feature <= threshold` / `feature > threshold`

---

## Re-Run Order

All stages from 01 onwards need re-running. The outputs in `outputs/` and
`models/` are stale (built on the old 9-feature set) and will be overwritten.

```
source .venv/Scripts/activate

python 01_baseline.py          # fast (~1 min)
python 02_conversion_uplift.py # fast (~2 min)
python 03_revenue_uplift.py    # SLOW — 3 causal forests, ~15-30 min
python 04_expected_value.py    # fast (~2 min)
python 05_feature_clustering.py # fast (~3 min)
python 06_cate_clustering.py   # fast (~2 min) -- no feature changes, but reads Stage 3 output
python 06b_recency_newbie.py   # fast (~1 min) -- model-free, re-run for completeness
python 07_policy_tree.py       # fast (~1 min)
python 07b_policy_tree_tuning.py # SLOW — 135 CV fits, ~10-20 min
```

Stage 3 and 7b are the long-running stages. Everything else is fast.

---

## What to Expect Downstream

The key question this re-run answers: does correcting the feature representation
change the CATE surface enough to affect policy recommendations?

Things that may change:
- CATE estimates for channel-differentiated customers (Multichannel may get
  stronger positive signal, Phone weaker)
- Womens email suppression for established customers may strengthen (the
  recency_x_newbie term gives the forest a direct handle on that interaction)
- Policy tree rules will now name `channel_Multichannel` and `zip_Urban`
  explicitly — more readable and defensible than "channel <= 1.5"
- CATE cluster sizes (C0 suppression group may grow or shrink)
- Policy value numbers ($0.71 rule-only, $1.10 C0-filtered) will update

Things that should NOT change:
- Overall ATE direction (Mens > Womens > Control)
- The suppression gap explanation (C0 defined by CATE scores, not feature
  boundaries — policy tree can only express 2,127 with clean rules regardless
  of feature encoding)
- The both-catalog data sparsity problem (encoding doesn't fix sample size)

After re-running, compare the new policy_tree_recommendations.csv and
07_policy_rules.txt to the values recorded in CLAUDE.md to assess impact.

---

## Next Task After Re-Run

Stage 8 — Reporting (`08_report.py`, not yet written).
See PLAN.md for spec. Two outputs: `report_technical.html` and
`report_stakeholder.html`.

Key narratives for the report:
1. Why feature representation matters — this re-run is a concrete example
2. The suppression gap (17,233 C0 vs 2,127 rules) and why it is not a failure
3. The two deployment tiers: rule-only (no model infra) vs C0-filtered (scoring pipeline)
4. Both-catalog segment: use RCT cross-tab averages, not model scores (data sparsity)
5. Limitations: spend-given-conversion instability (456/122 converters), channel x zip
   micro-segments not captured, both-catalog within-segment discrimination
