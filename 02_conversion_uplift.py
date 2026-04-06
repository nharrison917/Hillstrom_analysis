# -*- coding: utf-8 -*-
"""
Stage 2 - T-Learner Conversion Uplift Model
Target: binary conversion
Purpose: estimate who is persuaded to buy at all (incremental conversion probability)

T-Learner approach:
  - Train model_t on treated customers only
  - Train model_c on control customers only
  - Uplift score = model_t.predict_proba(X) - model_c.predict_proba(X)

Run three comparisons:
  A) Any email vs. No Email (collapsed)
  B) Mens Email vs. No Email
  C) Womens Email vs. No Email

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 02_conversion_uplift.py
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pickle

DATA_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
RANDOM_STATE = 42

FEATURES = [
    'recency',
    'log_history',
    'mens',
    'womens',
    'both_catalogs',
    'newbie',
    'history_segment_enc',
    'zip_code_enc',
    'channel_enc',
]

# ---------------------------------------------------------------------------
# 1. Load and prepare
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE)

df['log_history'] = np.log1p(df['history'])
df['both_catalogs'] = ((df['mens'] == 1) & (df['womens'] == 1)).astype(int)

cat_cols = ['history_segment', 'zip_code', 'channel']
le = LabelEncoder()
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

print(f"  Loaded {len(df)} rows.")
print()

# ---------------------------------------------------------------------------
# 2. T-Learner function
# ---------------------------------------------------------------------------
def train_t_learner(df_treated, df_control, features, target, label):
    """
    Train two XGBoost models (treated, control) and return both.
    Also reports a naive AUC on each subset as a sanity check.
    """
    print(f"--- T-Learner: {label} ---")
    print(f"  Treated n={len(df_treated)} | Control n={len(df_control)}")

    X_t = df_treated[features]
    y_t = df_treated[target]
    X_c = df_control[features]
    y_c = df_control[target]

    print(f"  Treated conversion rate: {y_t.mean():.4f}")
    print(f"  Control conversion rate: {y_c.mean():.4f}")

    spw_t = max(1, (y_t == 0).sum() / max(1, (y_t == 1).sum()))
    spw_c = max(1, (y_c == 0).sum() / max(1, (y_c == 1).sum()))

    model_t = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_t,
        eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0,
    )
    model_c = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_c,
        eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0,
    )

    # Train on each group's full data (T-Learner trains on subsets, not splits)
    model_t.fit(X_t, y_t, verbose=False)
    model_c.fit(X_c, y_c, verbose=False)

    print(f"  Models trained.")
    return model_t, model_c


def score_uplift(df_all, model_t, model_c, features):
    """
    Score all customers with both models.
    Uplift = P(convert | treated) - P(convert | control)
    """
    X = df_all[features]
    p_treat = model_t.predict_proba(X)[:, 1]
    p_ctrl  = model_c.predict_proba(X)[:, 1]
    uplift  = p_treat - p_ctrl
    return p_treat, p_ctrl, uplift


def qini_coefficient(df_scored, uplift_col, outcome_col, treatment_col):
    """
    Compute Qini coefficient — standard uplift model evaluation metric.
    Compares the uplift model's ranking against random targeting.
    Higher is better; 0 = random, 1 = perfect.
    """
    df_s = df_scored.copy().sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = len(df_s)
    n_t = df_s[treatment_col].sum()
    n_c = n - n_t

    cumulative_uplift = []
    cum_t_conv = 0
    cum_c_conv = 0
    cum_t = 0
    cum_c = 0

    for _, row in df_s.iterrows():
        if row[treatment_col] == 1:
            cum_t += 1
            cum_t_conv += row[outcome_col]
        else:
            cum_c += 1
            cum_c_conv += row[outcome_col]
        if cum_t > 0 and cum_c > 0:
            uplift_at_k = cum_t_conv / cum_t - cum_c_conv / cum_c
        else:
            uplift_at_k = 0
        cumulative_uplift.append(uplift_at_k * (cum_t + cum_c) / n)

    qini = np.trapezoid(cumulative_uplift) / n
    return qini, cumulative_uplift


# ---------------------------------------------------------------------------
# 3. Run comparisons
# ---------------------------------------------------------------------------
results = {}
score_cols = {}

# --- A: Any email vs. No Email ---
df_any_treated = df[df['segment'].isin(['Mens E-Mail', 'Womens E-Mail'])].copy()
df_any_treated['is_treated'] = 1
df_control = df[df['segment'] == 'No E-Mail'].copy()
df_control['is_treated'] = 0

mt_a, mc_a = train_t_learner(df_any_treated, df_control, FEATURES, 'conversion', 'Any Email vs Control')
p_t, p_c, uplift_a = score_uplift(df, mt_a, mc_a, FEATURES)
df['uplift_any'] = uplift_a
df['p_treat_any'] = p_t
df['p_ctrl_any'] = p_c
print()

# --- B: Mens Email vs. No Email ---
df_mens = df[df['segment'] == 'Mens E-Mail'].copy()
df_mens['is_treated'] = 1
df_ctrl2 = df_control.copy()

mt_b, mc_b = train_t_learner(df_mens, df_ctrl2, FEATURES, 'conversion', 'Mens Email vs Control')
_, _, uplift_b = score_uplift(df, mt_b, mc_b, FEATURES)
df['uplift_mens'] = uplift_b
print()

# --- C: Womens Email vs. No Email ---
df_womens = df[df['segment'] == 'Womens E-Mail'].copy()
df_womens['is_treated'] = 1

mt_c, mc_c = train_t_learner(df_womens, df_ctrl2, FEATURES, 'conversion', 'Womens Email vs Control')
_, _, uplift_c = score_uplift(df, mt_c, mc_c, FEATURES)
df['uplift_womens'] = uplift_c
print()

# ---------------------------------------------------------------------------
# 4. Qini curves
# ---------------------------------------------------------------------------
print("Computing Qini coefficients...")

df_any_eval = pd.concat([df_any_treated, df_control], ignore_index=True)
df_any_eval['uplift_any'] = df['uplift_any'].values[:len(df_any_eval)]

# Recompute on correct subsets
df_mens_eval = pd.concat([df_mens, df_ctrl2], ignore_index=True)
df_mens_eval['uplift_mens'] = score_uplift(df_mens_eval, mt_b, mc_b, FEATURES)[2]

df_womens_eval = pd.concat([df_womens, df_ctrl2], ignore_index=True)
df_womens_eval['uplift_womens'] = score_uplift(df_womens_eval, mt_c, mc_c, FEATURES)[2]

qini_a, curve_a = qini_coefficient(
    df[df['segment'].isin(['Mens E-Mail', 'Womens E-Mail', 'No E-Mail'])].assign(
        is_treated=lambda x: (x['segment'] != 'No E-Mail').astype(int)
    ),
    'uplift_any', 'conversion', 'is_treated'
)
qini_b, curve_b = qini_coefficient(
    df[df['segment'].isin(['Mens E-Mail', 'No E-Mail'])].assign(
        is_treated=lambda x: (x['segment'] == 'Mens E-Mail').astype(int)
    ),
    'uplift_mens', 'conversion', 'is_treated'
)
qini_c, curve_c = qini_coefficient(
    df[df['segment'].isin(['Womens E-Mail', 'No E-Mail'])].assign(
        is_treated=lambda x: (x['segment'] == 'Womens E-Mail').astype(int)
    ),
    'uplift_womens', 'conversion', 'is_treated'
)

print(f"  Qini - Any Email:    {qini_a:.6f}")
print(f"  Qini - Mens Email:   {qini_b:.6f}")
print(f"  Qini - Womens Email: {qini_c:.6f}")
print()

# ---------------------------------------------------------------------------
# 5. Uplift distribution plots
# ---------------------------------------------------------------------------
print("Building charts...")

fig_dist = make_subplots(
    rows=1, cols=3,
    subplot_titles=['Any Email Uplift', 'Mens Email Uplift', 'Womens Email Uplift']
)
for i, (col, label) in enumerate([
    ('uplift_any', 'Any'),
    ('uplift_mens', 'Mens'),
    ('uplift_womens', 'Womens'),
], 1):
    fig_dist.add_trace(
        go.Histogram(x=df[col], nbinsx=60, name=label, showlegend=False,
                     marker_color=['steelblue', '#EF553B', '#00CC96'][i-1]),
        row=1, col=i
    )
fig_dist.update_layout(
    title='Uplift Score Distributions (conversion probability lift)',
    template='plotly_white',
    height=400,
)
fig_dist.write_html('outputs/02_uplift_distributions.html')
print("  Saved: outputs/02_uplift_distributions.html")

# ---------------------------------------------------------------------------
# 6. Uplift by segment — who benefits most?
# ---------------------------------------------------------------------------
# Bin customers into uplift quartiles and check actual conversion rates
df['uplift_any_quartile'] = pd.qcut(df['uplift_any'], q=4, labels=['Q1 Low', 'Q2', 'Q3', 'Q4 High'])

treated_mask = df['segment'] != 'No E-Mail'
control_mask = df['segment'] == 'No E-Mail'

quartile_stats = []
for q in ['Q1 Low', 'Q2', 'Q3', 'Q4 High']:
    qmask = df['uplift_any_quartile'] == q
    t_conv = df[qmask & treated_mask]['conversion'].mean()
    c_conv = df[qmask & control_mask]['conversion'].mean()
    quartile_stats.append({
        'quartile': q,
        'treated_conversion': t_conv,
        'control_conversion': c_conv,
        'actual_uplift': t_conv - c_conv,
        'n_treated': (qmask & treated_mask).sum(),
        'n_control': (qmask & control_mask).sum(),
    })

qdf = pd.DataFrame(quartile_stats)
print()
print("=== Actual Conversion Rates by Predicted Uplift Quartile ===")
print(qdf.to_string(index=False))

fig_q = go.Figure()
fig_q.add_trace(go.Bar(
    x=qdf['quartile'], y=qdf['treated_conversion'] * 100,
    name='Treated', marker_color='steelblue'
))
fig_q.add_trace(go.Bar(
    x=qdf['quartile'], y=qdf['control_conversion'] * 100,
    name='Control', marker_color='lightgray'
))
fig_q.update_layout(
    title='Actual Conversion Rate by Predicted Uplift Quartile',
    xaxis_title='Predicted Uplift Quartile',
    yaxis_title='Conversion Rate (%)',
    barmode='group',
    template='plotly_white',
    height=450,
)
fig_q.write_html('outputs/02_uplift_by_quartile.html')
print("  Saved: outputs/02_uplift_by_quartile.html")

# ---------------------------------------------------------------------------
# 7. Feature importance comparison (treated vs control models)
# ---------------------------------------------------------------------------
imp_t = pd.Series(mt_a.feature_importances_, index=FEATURES).sort_values(ascending=True)
imp_c = pd.Series(mc_a.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig_fi = make_subplots(rows=1, cols=2, subplot_titles=['Treated Model', 'Control Model'])
fig_fi.add_trace(go.Bar(x=imp_t.values, y=imp_t.index, orientation='h',
                         marker_color='steelblue', showlegend=False), row=1, col=1)
fig_fi.add_trace(go.Bar(x=imp_c.values, y=imp_c.index, orientation='h',
                         marker_color='darkorange', showlegend=False), row=1, col=2)
fig_fi.update_layout(
    title='T-Learner Feature Importances: Treated vs Control Models',
    template='plotly_white', height=500,
)
fig_fi.write_html('outputs/02_feature_importance_comparison.html')
print("  Saved: outputs/02_feature_importance_comparison.html")

# ---------------------------------------------------------------------------
# 8. Save scores
# ---------------------------------------------------------------------------
scores_df = df[['recency', 'history', 'mens', 'womens', 'newbie',
                 'zip_code', 'channel', 'segment', 'conversion', 'spend',
                 'uplift_any', 'uplift_mens', 'uplift_womens',
                 'p_treat_any', 'p_ctrl_any', 'uplift_any_quartile']].copy()
scores_df.to_csv('outputs/scores_conversion.csv', index=False)
print("  Saved: outputs/scores_conversion.csv")

# ---------------------------------------------------------------------------
# 9. Save models and metadata
# ---------------------------------------------------------------------------
for name, model in [('mt_any', mt_a), ('mc_any', mc_a),
                     ('mt_mens', mt_b), ('mc_mens', mc_b),
                     ('mt_womens', mt_c), ('mc_womens', mc_c)]:
    with open(f'models/02_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

metadata = {
    'stage': 2,
    'model': 'T-Learner (XGBClassifier x2 per comparison)',
    'target': 'conversion',
    'features': FEATURES,
    'comparisons': ['any_email_vs_control', 'mens_email_vs_control', 'womens_email_vs_control'],
    'hyperparameters': {
        'n_estimators': 300, 'max_depth': 4,
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
    },
    'random_state': RANDOM_STATE,
    'qini_any': float(round(qini_a, 6)),
    'qini_mens': float(round(qini_b, 6)),
    'qini_womens': float(round(qini_c, 6)),
    'notes': (
        'T-Learner trains separate models on treated and control subsets. '
        'Uplift = P(convert|treated) - P(convert|control) applied to all customers. '
        'Qini coefficient is the standard uplift evaluation metric (higher=better, 0=random). '
        'Low Qini expected given 0.9% base conversion rate — sparse signal. '
        'Stage 3 causal forest on spend should produce more stable estimates.'
    )
}
with open('models/02_conversion_uplift_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("  Saved: models/02_conversion_uplift_metadata.json")

print()
print("=== Stage 2 complete ===")
