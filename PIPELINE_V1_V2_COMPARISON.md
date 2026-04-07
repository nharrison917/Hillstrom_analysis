# Pipeline Results: Label-Encoded (v1) vs One-Hot + Interaction (v2)

## What changed between v1 and v2

**v1 (9 features):** zip_code and channel were label-encoded as ordinal integers.
Tree models treated them as ordered: Urban > Suburban > Rural, Multichannel > Web > Phone.
That ordering is arbitrary and masks real response patterns.

**v2 (12 features):** zip_code and channel replaced with binary one-hot columns (Rural and
Phone as reference categories). A recency_x_newbie interaction term added explicitly.
See RERUN.md for full rationale.

---

## Stage 3 -- CATE Estimates

| Metric | v1 (label-enc, 9 feat) | v2 (one-hot + interaction, 12 feat) | Delta |
|--------|------------------------|--------------------------------------|-------|
| cate_any mean | $0.6402 | $0.6305 | -$0.010 |
| cate_any std | 2.1186 | 2.1087 | -0.010 |
| cate_mens mean | $0.7225 | $0.7178 | -$0.005 |
| cate_mens std | 1.9922 | 1.9592 | -0.033 |
| cate_womens mean | $0.4680 | $0.4626 | -$0.005 |
| cate_womens std | 2.0220 | 2.0129 | -0.009 |
| n negative CATE (any) | 15,074 | 15,020 | -54 |
| % positive CATE (any) | -- | 76.5% | -- |

**Interpretation:** CATE surface is essentially unchanged. The experiment-level signal
(ATE) is guaranteed by randomization, independent of feature representation. The slight
std reduction in cate_mens (-0.033) may indicate less spurious variance from the ordinal
encoding. No meaningful shift -- confirms the forest was already capturing the underlying
signal; v2 improves how it is expressed, not what it finds.

---

## Stage 6 -- CATE Clusters

### v1 (label-encoded, 9 features)
| Cluster | n | Persona | Mean CATE | Action |
|---------|---|---------|-----------|--------|
| C0 | 17,233 | Suppress -- Organic Buyers | -$0.59 | Do Not Email |
| C1 | 4,703 | Moderate Responders (conversion-driven) | +$0.57 | Send either |
| C2 | 38,588 | Moderate Responders (broad) | +$0.75 | Send Mens preferred |
| C3 | 3,476 | High Value Targets | +$5.58 | Priority send either |

### v2 (one-hot + interaction, 12 features)
| Cluster | n | Persona | Mean CATE | Action |
|---------|---|---------|-----------|--------|
| C0 | 16,889 | Suppress -- Marginal/Uncertain | -$0.64 | Do Not Email |
| C1 | 5,769 | High Value -- Conversion Driven | +$4.30 | Priority send either |
| C2 | 37,511 | Moderate Responders | +$0.64 | Send Mens preferred |
| C3 | 3,831 | Moderate Responders | +$0.59 | Send either |

**Key changes:**
- Suppression cluster (C0): -344 customers, CATE tightened from -$0.59 to -$0.64
- High-value cluster grew from 3,476 to 5,769 (+2,293 customers) but CATE diluted $5.58 -> $4.30.
  The cluster absorbed moderate-responders previously split differently.
- Cluster IDs are not stable across runs (KMeans assignment is arbitrary) -- compare by persona.
- Actual conversion in new high-value cluster (C1): 7.1% Mens, 6.9% Womens vs 0.05% control.

---

## Stage 7 -- Policy Tree Rules (untuned, depth=3)

### v1 -- suppression rules used label-encoded integer thresholds

The old tree expressed channel and zip as integer splits:
- `channel <= 1.5` (meaning: Phone or Web, not Multichannel -- but unstated)
- `zip_code_enc <= 0.5` (meaning: one zip category depending on alphabetical order)

Rules were mechanically correct but opaque. A stakeholder reading "channel <= 1.5" cannot
interpret it without the encoding map. Not defensible in a business presentation.

### v2 -- suppression rules use named columns and recency_x_newbie directly

**Any Email suppress** (n=2,139, CATE -$1.32):
```
history <= $30 AND recency > 6.5mo AND recency_x_newbie <= 3.5
```
Plain English: low-spend, lapsed, AND established (or barely-newbie) -> Do Not Email

**Mens Email suppress** (n=2,475, CATE -$1.08):
```
history <= $30 AND recency_x_newbie <= 0.50 AND recency > 5.5mo
```
`recency_x_newbie <= 0.5` cleanly isolates established customers:
newbie=0 means interaction=0 regardless of recency value.

**Womens Email suppress** (n=8,397, CATE -$0.26):
```
recency_x_newbie <= 0.50 AND history $213-$385
```
Established mid-spend customers show near-zero/negative Womens response.
This leaf is much larger than v1 equivalent -- interaction term makes the split cleaner.

**Notable send leaves in v2:**
- Mens: `zip_Urban = 1 AND recency <= 5.5mo AND established` -> CATE $2.09
- Any: `channel_Multichannel = 1 AND history > $257 AND womens = 1` -> CATE $1.17

These column names appear by name because one-hot encoding made them explicit features.

---

## Stage 7b -- Tuned Policy Tree

| Metric | v1 (label-enc) | v2 (one-hot + interaction) | Notes |
|--------|----------------|---------------------------|-------|
| Any Email params | depth=4, min_leaf=500 | depth=4, min_leaf=500 | Identical |
| Mens Email params | depth=4, min_leaf=1000 | depth=4, min_leaf=200 | Smaller leaf; cleaner signal |
| Womens Email params | depth=4, min_leaf=500 | depth=3, min_leaf=200 | Depth reduced -- less overfit |
| Tuned tree DNE (rule-only) | 2,127 | 0 | Suppression split unstable in CV |
| Untuned tree DNE | 2,127 | ~2,127 | Same sharp boundary persists |
| Rule-only policy value | $0.71 | $0.63 | See note below |
| C0-filtered policy value | $1.10 | $1.09 | Essentially unchanged |
| C0 cluster DNE count | 17,831 | 16,889 | C0 cluster shrank by 344 |
| Leakage fixed by C0 filter | 292 | 1,743 | Womens leaf is larger in v2 |

**On the rule-only policy value drop ($0.71 -> $0.63):**
Not a regression. In v1, tuned tree found 2,127 DNE customers (worst-CATE segment),
raising the average of the remaining sent group to $0.71. In v2 the CV-selected tree
sends everyone (0 DNE), so rule-only = full-population mean = $0.63. The C0-filtered
deployment ($1.09) is the right comparison for any suppression strategy.

---

## Summary: What the Feature Engineering Actually Achieved

**Did NOT meaningfully change:**
- Overall CATE estimates (ATEs within sampling noise)
- Suppression cluster size (C0: 17,233 -> 16,889, -2%)
- C0-filtered deployment value ($1.10 -> $1.09)
- Core finding: lapsed newbies = best Mens; established = weak Womens

**DID change (the interpretability wins):**
- Policy rules now name features humans can read without a lookup table
- `recency_x_newbie` appears directly in suppression rules
- `channel_Multichannel` and `zip_Urban` appear by name in high-value send leaves
- Womens tree depth reduced 4 -> 3 (CV more stable, less overfitting)
- Mens min_leaf reduced 1000 -> 200 (cleaner interaction term allows finer splits)
- Leakage fixed increased 292 -> 1,743 (larger Womens suppress leaf)

The feature engineering was the right call for interpretability and portfolio presentation.
The causal signal was already there; v2 surfaces it in language a business audience can read.
