# Hillstrom Uplift Modeling Project

## Dataset
`Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`

64,000 rows. A randomized email marketing experiment from Kevin Hillstrom (2008).
Widely used as a benchmark for uplift / causal modeling.

### Key Fields
| Field | Description |
|-------|-------------|
| `recency` | Months since last purchase |
| `history_segment` | Categorical spend tier |
| `history` | Actual $ spend in prior year |
| `mens`, `womens` | Binary: purchased from mens/womens catalog |
| `zip_code` | Urban / Suburban / Rural |
| `newbie` | 1 = new customer in last 12 months |
| `channel` | Phone / Web / Multichannel |
| `segment` | Treatment assignment: Mens Email / Womens Email / No Email |
| `visit`, `conversion` | Binary outcomes post-campaign |
| `spend` | Dollar outcome post-campaign (zero-inflated, right-skewed) |

### Treatment Groups
- Mens Email (treatment A)
- Womens Email (treatment B)
- No Email (control)

Note: 6,448 customers have both `mens=1` and `womens=1` (purchased from both catalogs).

---

## Business Question
Which customers should receive an email, and which type?

Standard conversion uplift is not enough. A customer who rarely converts but
spends $1,000 when they do is more valuable than one who converts at 2x the rate
but only spends $10. The target metric is **expected incremental revenue per
customer** from a given email send.

---

## Modeling Plan

### Stage 1 — Baseline (understand the data)
- File: `01_baseline.py`
- Model: XGBoost classifier on `conversion` (binary)
- Purpose: feature importances, sanity check on data, confirm preprocessing works
- Output: `outputs/01_feature_importance.html`

### Stage 2 — Conversion Uplift
- File: `02_conversion_uplift.py`
- Model: T-Learner (two XGBoost models: one on treated, one on control)
- Target: binary `conversion`
- Purpose: who is persuaded to buy at all
- Outputs: `outputs/02_uplift_conversion.html`, uplift scores saved to `outputs/scores_conversion.csv`

### Stage 3 — Revenue Uplift (Causal Forest)
- File: `03_revenue_uplift.py`
- Model: Causal Forest (via `econml`) with `spend` as continuous outcome
- Purpose: estimate CATE on dollars — incremental revenue per customer
- Handles zero-inflation implicitly; causal forest is robust enough for first pass
- Outputs: `outputs/03_cate_distribution.html`, CATE scores saved to `outputs/scores_revenue.csv`

### Stage 4 — Joint Expected Value Model
- File: `04_expected_value.py`
- Combines Stage 2 and Stage 3:
  `EV = P(converts | email) * E(spend | converts, email) - P(converts | control) * E(spend | converts, control)`
- Rank customers by EV to produce a prioritized send list
- Outputs: `outputs/04_ev_ranking.html`, `outputs/send_list.csv`

### Stage 5 — Feature-Space Clustering + Lift Mapping
- File: `05_feature_clustering.py`
- K-means on customer attributes; elbow/silhouette to choose k
- Map each cluster onto lift space (x=conversion lift, y=spend lift, bubble=size)
- Purpose: do natural customer groups separate in response space? Persona labels.
- Outputs: `outputs/05_feature_clusters.html`, `outputs/05_clusters_lift_map.html`

### Stage 6 — CATE-Space Clustering
- File: `06_cate_clustering.py`
- Cluster on response score vectors: [cate_any, cate_mens, cate_womens,
  uplift_conv_any, uplift_conv_mens, uplift_conv_womens, ev_conv_component, ev_spend_component]
- Profile clusters back against original features to explain who each type is
- Purpose: response typology — suppression, mens-specific, womens-specific, etc.
- Outputs: `outputs/06_cate_clusters.html`, `outputs/06_cate_cluster_profiles.html`

### Stage 6b — Recency/Newbie Exploration (DONE)
- File: `06b_recency_newbie.py`
- Cross-tab of recency buckets x newbie x email type x actual outcomes (RCT-valid causal estimates)
- Key finding: interaction between recency and newbie drives Womens email suppression.
  Established customers barely respond to Womens email (+$0.11 overall).
  Lapsed newbies (10-12mo) are highest Mens responders (+$1.26 lift).
- Outputs: `outputs/06b_recency_lift.html`, `outputs/06b_recency_newbie_heatmap.html`,
  `outputs/06b_newbie_lift.html`

### Stage 7b — Policy Tree Tuning (DONE)
- File: `07b_policy_tree_tuning.py`
- 5-fold CV (stratified on treatment group), 27 combinations:
  depth [2,3,4] x min_samples_leaf [200,500,1000] x min_impurity_decrease [0,0.01,0.05]
- Also tested conservative lower-bound training target (cate_any_lb) — sent 0% (CIs too wide)
- Best params: Any=depth4/min500, Mens=depth4/min1000 (actual depth 2), Womens=depth4/min500
- C0 upstream filter added: applying Stage 6 cluster C0 as suppression gate raises
  policy value $0.71 → $1.10/customer (+56%). Both versions in policy_tree_recommendations.csv.
- Outputs: `outputs/07b_tuning_heatmap_*.html`, `outputs/07_policy_rules.txt`

### Stage 7 — Policy Tree (original, untuned)
- File: `07_policy_tree.py`
- econml PolicyTree on CATE estimates — learns shallow decision rules
  that maximize expected incremental revenue
- Target depth: 3-4 (interpretable, deployable without scoring pipeline)
- Outputs: `outputs/07_policy_tree.html`, `outputs/07_policy_rules.txt`

### Stage 8 — Reporting
- File: `08_report.py`
- Stakeholder-facing HTML summary (non-technical, uses cluster personas)
- Technical HTML with model details, limitations, methodology notes
- Incorporates findings from channel, zip, both-catalog analysis
- References POSSIBLE_DIRECTIONS.md for limitations section
- Outputs: `outputs/report_technical.html`, `outputs/report_stakeholder.html`

---

## Treatment Handling
Stages 2-4 should be run twice:
- Any email vs. no email (collapsed binary)
- Mens email vs. no email
- Womens email vs. no email

The collapsed analysis establishes the baseline; the split analysis reveals
whether the email type matters and for whom.

---

## Environment
- Python 3.11.8
- `.venv` in project root (activate before running anything)
- Windows: `.venv\Scripts\activate`
- Git Bash: `source .venv/Scripts/activate`

## Key Packages Needed
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `econml` (Microsoft's causal ML library — installs heavy deps, do separately)
- `plotly`
- `shap` (optional, for feature explanation in Stage 3+)

Install order matters: install `econml` last as it pulls in specific numpy/scipy versions.

---

## Output Conventions
- All scripts save `.html` visualizations to `outputs/`
- All intermediate scores saved to `outputs/` as `.csv`
- Models saved to `models/` as `.pkl`
- Model metadata (hyperparameters, thresholds, feature lists) saved to `models/` as `.json`
- `random_state=42` everywhere

---

## Status
| Stage | Status |
|-------|--------|
| Environment setup | Done — .venv created |
| Data exploration | Done (in chat — see notes below) |
| Stage 1 baseline | Done |
| Stage 2 conversion uplift | Done |
| Stage 3 revenue uplift | Done |
| Stage 4 expected value | Done |
| Stage 5 feature clustering | Done — k=3, catalog-type dominated |
| Stage 6 CATE clustering | Done — k=4, see CLAUDE.md for cluster summary |
| Stage 6b recency/newbie exploration | Done — key: recency x newbie interaction |
| Stage 7 policy tree | Done (superseded by 7b) |
| Stage 7b policy tree tuning | Done — 5-fold CV, C0 filter, rules exported |
| Stage 8 reporting | NOT STARTED — next task |

## Data Exploration Notes (pre-modeling)
- `history`: mean=$242.09, median=$158.11, mode=$29.99, min=$29.99, max=$3,345.93
  - Right-skewed. $29.99 appears to be a minimum purchase threshold.
  - Consider log-transforming `history` as a feature.
- `spend` is zero-inflated: most customers have spend=0 post-campaign.
- 6,448 records have both `mens=1` and `womens=1` — overlap segment, handle explicitly.

---

## Session Notes
- Project framing discussed in initial session (2026-04-05).
- Decision to model spend (not just conversion) made explicitly — the business
  case is expected incremental revenue, not conversion rate alone.
- Tableau was the original tool; ML chosen because too many interactions to
  surface manually in a viz tool.
