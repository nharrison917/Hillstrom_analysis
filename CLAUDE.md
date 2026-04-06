# Hillstrom Uplift Modeling — Agent Context

Read this file first at the start of every session.
Also read PLAN.md for stage status and POSSIBLE_DIRECTIONS.md for known blind spots.

---

## What This Project Is

A full causal ML pipeline on the Hillstrom e-mail marketing dataset (64,000 customers,
randomized experiment: Mens Email / Womens Email / No Email control).

**Business question:** Which customers should receive an email, and which type?
Not just "who converts" — but "who generates incremental revenue *because of* the email."
Customers who would have bought anyway should be suppressed, not targeted.

**Key insight already established:** Negative uplift is real. ~17,000 customers
(CATE cluster C0) show negative expected value from emailing. ~2,127 have a clean
rule-based suppression (policy tree). The gap between these two numbers is documented
in POSSIBLE_DIRECTIONS.md item 7.

---

## Environment

- Python 3.11.8, `.venv` in project root
- Activate: `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\activate` (PowerShell)
- Key packages: pandas, numpy, scikit-learn, xgboost, econml, plotly
- All scripts run from project root: `python 0X_scriptname.py`
- Windows 11, VS Code

---

## Dataset

`Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`

64,000 rows. Key fields:
- `recency`: months since last purchase
- `history`: prior year $ spend (mean=$242, median=$158, mode=$29.99 — right-skewed, use log_history as feature)
- `mens`, `womens`: binary catalog purchase flags (6,448 customers have both=1)
- `zip_code`: Urban / Suburban / Rural
- `newbie`: 1 = new customer in last 12 months
- `channel`: Phone / Web / Multichannel
- `segment`: treatment assignment (Mens E-Mail / Womens E-Mail / No E-Mail)
- `visit`, `conversion`: binary outcomes
- `spend`: dollar outcome (zero-inflated — 0.9% conversion rate, 578 converters)

---

## Standard Feature Set (used in all models)

```python
FEATURES = [
    'recency', 'log_history', 'mens', 'womens', 'both_catalogs', 'newbie',
    'history_segment_enc', 'zip_code_enc', 'channel_enc',
]
```

Always add these derived columns before modeling:
```python
df['log_history']   = np.log1p(df['history'])
df['both_catalogs'] = ((df['mens'] == 1) & (df['womens'] == 1)).astype(int)
le = LabelEncoder()
for col in ['history_segment', 'zip_code', 'channel']:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
```

`random_state=42` everywhere.

---

## Pipeline Stages — Status

| Stage | File | Status | Key Output |
|-------|------|--------|------------|
| 1 - Baseline XGBoost | `01_baseline.py` | Done | ROC-AUC 0.55 (intentionally weak — treatment excluded) |
| 2 - T-Learner conversion uplift | `02_conversion_uplift.py` | Done | `outputs/scores_conversion.csv` |
| 3 - Causal Forest on spend | `03_revenue_uplift.py` | Done | `outputs/scores_revenue.csv` |
| 4 - Expected value ranking | `04_expected_value.py` | Done | `outputs/send_list.csv` |
| 5 - Feature clustering | `05_feature_clustering.py` | Done | k=3, catalog-type dominated |
| 6 - CATE clustering | `06_cate_clustering.py` | Done | k=4, see cluster summary below |
| 6b - Recency/newbie exploration | `06b_recency_newbie.py` | Done | `outputs/06b_recency_newbie_heatmap.html` |
| 7 - Policy tree | `07_policy_tree.py` | Done (superseded by 7b) | `outputs/07_policy_rules.txt` |
| 7b - Policy tree tuning | `07b_policy_tree_tuning.py` | Done | `outputs/07b_tuning_heatmap_*.html` |
| 8 - Reporting | `08_report.py` | NOT STARTED | **Next task** |

---

## Key Findings To Date

### Treatment group baselines
- Mens email: 1.25% conversion, $1.42 mean spend
- Womens email: 0.88% conversion, $1.08 mean spend
- Control: 0.57% conversion, $0.65 mean spend

### CATE clusters (Stage 6) — the most important segmentation
| Cluster | n | Persona | Mean CATE | Action |
|---------|---|---------|-----------|--------|
| C0 | 17,233 | Suppress — Organic Buyers | -$0.59 | Do Not Email |
| C1 | 4,703 | Moderate Responders (conversion-driven) | +$0.57 | Send either |
| C2 | 38,588 | Moderate Responders (broad) | +$0.75 | Send Mens preferred |
| C3 | 3,476 | High Value Targets | +$5.58 | Priority send either |

C3 profile: mean history $396, Urban skew, 19% both-catalog. Actual conversion
9.25% (Mens), 8.80% (Womens) vs 0.09% control. Spend lift ~$17.79.

### Policy tree rules (Stage 7, depth=3, min_samples_leaf=500)
Saved in full at `outputs/07_policy_rules.txt`. Key suppressions:
- Mens suppress: history <= $30 AND not newbie AND recency > 5.5 months (-$1.04 CATE)
- Womens suppress: Phone/Web AND history <= $30 AND recency > 6.5 months (-$0.08 CATE)
- True DNE (both suppressed): n=2,127. Profile: 100% Phone/Web, history=$29.99,
  recency=9.45 months, 0% newbie, 0% Multichannel.

### Stage 6b findings — Recency x Newbie (new)
The key interaction is recency x newbie, not recency alone. All recency buckets show
positive lift, but splitting by newbie status reveals:
- Recent (1-2mo) established customers: Womens email lift = **-$0.21** (suppress)
- Lapsing (7-9mo) established customers: Womens email lift = **-$0.14** (suppress)
- Lapsed (10-12mo) newbies: Mens lift = **+$1.26** (highest in dataset — win-back scenario)
- Newbie + both_catalogs + Multichannel (n=1,372): Mens +$1.89, Womens +$2.04 (priority)
Established customers overall barely respond to Womens email (+$0.11 lift).
Full breakdown in `outputs/06b_recency_newbie_heatmap.html` and NOTES.md.

### C0 upstream suppression filter (Stage 7b addition)
Applying the CATE C0 cluster as an upstream gate (if cate_cluster==0 → Do Not Email):
- Rule-only policy: 2,127 DNE, mean policy value of sent customers = **$0.71**
- C0-filtered policy: 17,831 DNE, mean policy value of sent customers = **$1.10** (+56%)
Both versions saved in `outputs/policy_tree_recommendations.csv`
(columns: `policy_recommendation` = rule-only, `policy_c0_filtered` = with C0 gate).

### Known blind spots (see POSSIBLE_DIRECTIONS.md for full details)
1. Channel x email responsiveness — Multichannel lift 2x single-channel, model misses it
2. Both-catalog segment — 2.5x lift, too small for causal forest to discriminate within
3. Zip x email type — Suburban responds to Mens, Urban to Womens; model flat on zip
4. Channel x Zip micro-segments — Rural Multichannel highest conversion, no interaction feature
5. Spend-given-conversion instability — only 122-456 converters per group
6. Recency/Newbie interactions — **EXPLORED** (Stage 6b). Key: established customers
   don't respond to Womens email; lapsed newbies are highest Mens responders.
7. Policy tree suppression leakage — **RESOLVED** (Stage 7b). C0 filter implemented.
   292 leakage customers fixed. See C0 filter results above.

---

## Stage 7b Results (Policy Tree Tuning)

**Best parameters (selected by 5-fold CV policy value):**
- Any Email: depth=4, min_samples_leaf=500, min_impurity_decrease=0.0
- Mens Email: depth=4, min_samples_leaf=1000, min_impurity_decrease=0.0 (actual depth=2)
- Womens Email: depth=4, min_samples_leaf=500, min_impurity_decrease=0.0

**Key findings from tuning:**
1. Conservative lower-bound target suppressed 100% of customers (sent 0%) — CIs too wide.
   Mean cate_any_lb = -$1.63. Only 578 converters leaves the forest unable to be confident
   anyone has positive CATE. Real signal: the point estimates are uncertain, not wrong.
2. Any min_impurity_decrease > 0 collapses all trees to depth=0 (100% sent, no splits).
   Every split's policy value gain is < $0.01. The CATE surface is shallow across features.
3. Suppression gap unchanged: DNE still 2,127. C0 customers (17,233) scatter across feature
   space — no depth-4 rule can cleanly separate them. The rule-based suppression captures
   customers at a sharp boundary (history=$29.99, old, non-newbie), not the full C0 population.
4. Mens tree collapsed to actual depth=2 despite depth=4 request — confirms depth=2 is right.
5. Womens tree unstable at depth=3-4 (std=0.11 across folds) — CV splits sometimes find
   a suppression leaf, sometimes don't. Signals overfitting to the noisy CATE surface.

**Suppression gap explanation (now understood):**
The 17,233 vs 2,127 gap is NOT a failure of tuning. The C0 cluster is defined by CATE
scores (model output), not by feature boundaries. Policy tree can only express rules in
feature space. The 2,127 with clean rules sit at a sharp boundary; the remaining ~15,000
need the Stage 4 send list (continuous score) to suppress, not a rule. This is the
right narrative for Stage 8 reporting.

**Leakage (Blind Spot #7) — RESOLVED:** C0 upstream filter applied in Stage 7b.
292 leakage customers fixed. Both policy versions in `outputs/policy_tree_recommendations.csv`.

---

## Output Conventions

- HTML visualisations → `outputs/`
- Score CSVs → `outputs/`
- Trained models → `models/*.pkl`
- Model metadata → `models/*_metadata.json`
- All scripts self-contained and runnable from project root with .venv active
