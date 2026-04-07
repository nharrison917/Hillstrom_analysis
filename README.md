# Hillstrom Uplift Modeling

**Causal ML for smarter email targeting: identifying who to send, who to suppress, and why.**
Estimating individual-level causal lift to maximize incremental revenue -- not just conversion.

> Live app: [hillstromanalysis-fx6nmhe3qw9kepdfuvgmpj.streamlit.app](https://hillstromanalysis-fx6nmhe3qw9kepdfuvgmpj.streamlit.app/)

---

## The Problem

Standard conversion models answer the wrong question. A model trained to predict
"who will buy?" will confidently score loyal customers who were always going to purchase,
wasting campaign spend and, in some cases, actively harming revenue by interfering
with organic purchases.

This project answers a harder question: **which customers buy *more because of* the
email?** Using a 2008 randomized controlled trial (RCT) dataset, the pipeline estimates
the Conditional Average Treatment Effect (CATE), the individual-level causal lift,
for three treatment arms: Mens Email, Womens Email, and No Email control.

The business implication: not emailing 16,889 customers improves estimated campaign
revenue by **+27% ($40,320 → $51,351)** compared to a blanket send.

---

## Dataset

[Kevin Hillstrom's MineThatData E-Mail Analytics challenge](http://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) (2008).
64,000 customers, randomized across three groups in equal proportions.

| Field | Description |
|-------|-------------|
| `recency` | Months since last purchase (1–12) |
| `history` | Prior-year spend in dollars (mean $242, median $158, right-skewed) |
| `mens`, `womens` | Binary: purchased from mens/womens catalog |
| `newbie` | 1 = new customer in last 12 months |
| `channel` | Phone / Web / Multichannel |
| `zip_code` | Urban / Suburban / Rural |
| `segment` | Treatment: Mens E-Mail / Womens E-Mail / No E-Mail |
| `spend` | Dollar spend post-campaign (zero-inflated; 0.9% conversion rate) |

Treatment group baselines (actual RCT outcomes):

| Group | Conversion rate | Mean spend |
|-------|-----------------|------------|
| Mens Email | 1.25% | $1.42 |
| Womens Email | 0.88% | $1.08 |
| No Email (control) | 0.57% | $0.65 |

---

## Pipeline Overview

Eight stages, each a self-contained Python script, building from raw data to a
deployable targeting policy with an interactive dashboard.

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 - Baseline | `01_baseline.py` | XGBoost classifier on conversion; feature importance sanity check |
| 2 - Conversion Uplift | `02_conversion_uplift.py` | T-Learner estimating P(converts\|email) − P(converts\|control) |
| 3 - Revenue Uplift | `03_revenue_uplift.py` | Causal Forest (EconML) estimating CATE on dollar spend |
| 4 - Expected Value | `04_expected_value.py` | Combines stages 2 and 3 into a per-customer EV score; ranked send list |
| 5 - Feature Clustering | `05_feature_clustering.py` | K-means on customer attributes; lift mapping per cluster |
| 6 - CATE Clustering | `06_cate_clustering.py` | Clusters customers by *response profile*: the actionable segmentation |
| 6b - Recency x Newbie | `06b_recency_newbie.py` | RCT cross-tab to characterize the strongest interaction in the data |
| 7 - Policy Tree | `07_policy_tree.py` | Shallow decision tree on CATE estimates; human-readable targeting rules |
| 7b - Policy Tuning | `07b_policy_tree_tuning.py` | 5-fold CV over tree hyperparameters; C0 upstream suppression gate |
| 8 - Dashboard | `08_app.py` | Streamlit app: stakeholder and technical views |

### Feature engineering (`preprocessing.py`)

All stages share a single feature module. Final feature set (12 features):

```python
FEATURES = [
    'recency', 'log_history', 'mens', 'womens', 'both_catalogs', 'newbie',
    'history_segment_enc',   # ordinal spend tier (genuine ordering)
    'zip_Suburban',          # one-hot (Rural = reference, weakest email responder)
    'zip_Urban',
    'channel_Multichannel',  # one-hot (Phone = reference, weakest email responder)
    'channel_Web',
    'recency_x_newbie',      # explicit interaction: recency * newbie
]
```

The `recency_x_newbie` interaction term was added after Stage 6b revealed it as the
strongest effect modifier in the dataset. One-hot encoding for zip/channel replaced
label encoding after label encoding imposed false ordinal structure that masked the
Suburban/Urban email-type flip.

---

## Key Findings

### 1. Not all customers benefit from email

Clustering customers by their CATE response profile (Stage 6) produced four segments
with meaningfully different implications for targeting:

| Cluster | n | Persona | Mean CATE | Action |
|---------|---|---------|-----------|--------|
| C0 | 16,889 | Suppress: marginal or negative responders | -$0.64 | Do Not Email |
| C1 | 5,769 | High value: conversion-driven | +$4.30 | Priority send (either type) |
| C2 | 37,511 | Moderate responders | +$0.64 | Send; Mens preferred |
| C3 | 3,831 | Moderate responders | +$0.59 | Send either |

C0 customers are not just unresponsive; they show *negative* expected value from
emailing. Suppressing them raises mean policy value from $0.63 to $1.09 per customer
sent (+73% among the customers who actually receive an email).

C1 profile: mean history $384, 59% newbie, 21% hold both catalogs. Actual conversion
rates 7.1% (Mens) and 6.9% (Womens) vs. 0.05% for C1 customers in the control group.

### 2. The recency × newbie interaction is the strongest effect modifier

`recency_x_newbie = recency * newbie` encodes a behavioral insight: the value of
recency depends entirely on whether the customer is new.

RCT cross-tab results (Stage 6b):

| Segment | Mens lift | Womens lift |
|---------|-----------|-------------|
| Recent (1–2 mo), established | +$0.41 | **−$0.21** (suppress) |
| Lapsing (7–9 mo), established | +$0.35 | **−$0.14** (suppress) |
| Lapsed (10–12 mo), newbie | **+$1.26** (highest) | +$0.49 |
| Newbie + both catalogs + Multichannel | **+$1.89** | **+$2.04** |

Established customers barely respond to Womens email (+$0.11 overall). Lapsed newbies
are the best Mens email targets (the win-back dynamic). This interaction appears
directly in the learned policy tree rules, replacing opaque thresholds from the v1
label-encoded feature set.

### 3. Two-tier suppression policy outperforms blanket send by 27%

Three targeting strategies compared on estimated revenue across 64,000 customers:

| Strategy | Customers emailed | Estimated revenue |
|----------|-------------------|-------------------|
| Blanket send | 64,000 (100%) | $40,320 |
| Rule-only (policy tree) | ~61,900 (96.7%) | $43,311 |
| C0-filtered (CATE gate + rules) | 47,111 (73.6%) | $51,351 |

The C0 cluster filter is the primary driver. The ~2,127 rule-based suppressions from
the depth-3 policy tree represent customers at a sharp feature boundary (history ≤ $30,
lapsed, established); the remaining ~14,800 C0 suppressions require the continuous CATE
score, which is why the tree alone captures only a fraction of the full suppression gain.

### 4. Policy tree rules are human-readable and deployable

Selected rules from the tuned policy tree (Stage 7b):

```
# Any email - suppress:
IF history <= $30 AND recency > 6.50 AND recency_x_newbie <= 3.50
  -> DO NOT EMAIL  (mean CATE: -$1.73 to -$0.94 depending on mens catalog)

# Mens email - suppress:
IF history <= $30 AND recency_x_newbie <= 0.50 AND recency > 7.50
  -> DO NOT EMAIL  (mean CATE: -$1.21)

# Womens email - suppress:
IF recency_x_newbie <= 0.50 AND history $213-$385
  -> DO NOT EMAIL  (mean CATE: -$0.26)
```

The `recency_x_newbie <= 0.50` threshold cleanly isolates established customers
(`newbie=0` produces an interaction of 0, below any positive threshold).

---

## How to Run Locally

**Requirements:** Python 3.11, Git

```bash
git clone <repo-url>
cd hillstrom

# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Git Bash
# or: .venv\Scripts\activate    # PowerShell

# Install dependencies
python -m pip install -r requirements.txt

# Run any pipeline stage
python 01_baseline.py
python 02_conversion_uplift.py
# ... and so on through 07b_policy_tree_tuning.py

# Launch dashboard
streamlit run 08_app.py
```

The data CSV must be present in the project root for the recency heatmap to render.
All other dashboard components load from pre-computed `outputs/` files.

---

## Dashboard

The Streamlit app (`08_app.py`) has two audience modes, toggled in the sidebar:

**Stakeholder view:** plain-language findings with four cluster cards (personas,
estimated revenue impact per cluster), a three-way policy comparison chart (blanket vs.
rule-only vs. C0-filtered), and the recency × newbie interaction heatmap with callout
annotations.

**Technical view:** four tabs covering CATE distribution curves with quartile
validation, full policy tree rules text with tuning results, feature importance charts
from both the causal forest (Stage 3) and XGBoost baseline (Stage 1), and a methodology
and limitations section.

---

## Limitations

1. **CATE estimates, not ground truth.** The causal forest estimates individual lift from
   a 64,000-row dataset split across three treatment arms. Point estimates carry wide
   confidence intervals, particularly for small subgroups (both-catalog + Multichannel
   newbies, n≈1,372).

2. **Outcome instability.** Only 578 customers converted across all groups. CATE
   estimates for spend-given-conversion are noisy by design; this is a property of the
   dataset, not the model.

3. **The conservative lower-bound policy sends 0%.** When the policy tree is trained on
   the CATE lower confidence bound rather than the point estimate, it produces a depth-0
   tree that suppresses everyone. This is the correct behavior: confidence intervals on a
   0.9% conversion outcome are wide enough that the model cannot confidently assign
   positive CATE to any customer. A policy that refuses to act under genuine uncertainty
   is working as intended. The deployed policy uses point estimates, with the lower-bound
   result documented as a calibration check.

4. **Within-both-catalog discrimination is flat.** The causal forest cannot reliably
   discriminate within the 6,448 both-catalog customers (EV spread Q1 $1.81 vs. Q4
   $1.95). This is a data sparsity problem. The proposed fix (Iteration B: RCT group
   average override for this segment) was not implemented in this version.

5. **External validity.** The Hillstrom dataset is a 2008 e-commerce benchmark. Channel
   mix, customer behavior, and email response rates will differ substantially in modern
   contexts. The pipeline design is generalizable; the specific thresholds are not.

---

## Next Steps

The revenue lift reported here is estimated from CATE model scores, not validated against
a prospective holdout. A real deployment would run a randomized A/B test -- C0-filtered
policy vs. blanket send -- to confirm the $11k gain on fresh data and measure the true
suppression effect. With a larger observation window or higher campaign volume, confidence
intervals on per-customer CATE estimates would narrow, which would directly improve the
reliability of the suppression boundary and reduce the gap between the point-estimate and
lower-bound policies.

---

## Project Structure

```
hillstrom/
├── preprocessing.py              # Shared feature engineering module
├── 01_baseline.py                # Stage 1: XGBoost baseline
├── 02_conversion_uplift.py       # Stage 2: T-Learner conversion uplift
├── 03_revenue_uplift.py          # Stage 3: Causal Forest CATE
├── 04_expected_value.py          # Stage 4: EV scoring and ranking
├── 05_feature_clustering.py      # Stage 5: Feature-space K-means
├── 06_cate_clustering.py         # Stage 6: Response-profile clustering
├── 06b_recency_newbie.py         # Stage 6b: Recency x newbie RCT cross-tab
├── 07_policy_tree.py             # Stage 7: Policy tree (untuned baseline)
├── 07b_policy_tree_tuning.py     # Stage 7b: CV tuning + C0 suppression gate
├── 08_app.py                     # Stage 8: Streamlit dashboard
├── models/                       # Trained models (.pkl) and metadata (.json)
├── outputs/                      # CSVs, policy rules, HTML visualizations
├── .streamlit/config.toml        # App theme (Inter font, Tailwind palette)
├── requirements.txt
└── PLAN.md                       # Full design notes and stage-by-stage decisions
```

---

*This project was built to demonstrate applied causal ML for marketing optimization.
Claude Code (Anthropic) was used as an AI coding assistant throughout. All modeling
decisions, interpretations, and limitations are documented in PLAN.md and CLAUDE.md.*
