# -*- coding: utf-8 -*-
"""
Stage 3 - Causal Forest on Spend (CATE estimation)
Target: continuous spend (dollars)
Purpose: estimate incremental revenue per customer from each email type

CausalForestDML from econml estimates CATE directly:
  CATE(x) = E[spend | do(email), X=x] - E[spend | do(no email), X=x]

This answers: "how many MORE dollars does this customer spend because of the email?"

Run three comparisons (same structure as Stage 2):
  A) Any email vs. No Email
  B) Mens Email vs. No Email
  C) Womens Email vs. No Email

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 03_revenue_uplift.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import CausalForestDML
import pickle

warnings.filterwarnings('ignore')

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

le = LabelEncoder()
for col in ['history_segment', 'zip_code', 'channel']:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

print(f"  Loaded {len(df)} rows.")
print(f"  Spend > 0: {(df['spend'] > 0).sum()} rows ({(df['spend'] > 0).mean():.2%})")
print(f"  Mean spend (all): ${df['spend'].mean():.4f}")
print(f"  Mean spend (converters only): ${df[df['spend'] > 0]['spend'].mean():.2f}")
print()

# ---------------------------------------------------------------------------
# 2. Causal Forest helper
# ---------------------------------------------------------------------------
def run_causal_forest(df_subset, treatment_col, features, outcome, label):
    """
    Fit a CausalForestDML model on df_subset.
    treatment_col: binary 0/1 column indicating treatment assignment.
    Returns the fitted model and CATE estimates for df_subset.
    """
    print(f"--- Causal Forest: {label} ---")
    print(f"  n={len(df_subset)} | Treated={df_subset[treatment_col].sum()} | Control={(df_subset[treatment_col]==0).sum()}")
    print(f"  Treated mean spend: ${df_subset[df_subset[treatment_col]==1][outcome].mean():.4f}")
    print(f"  Control mean spend: ${df_subset[df_subset[treatment_col]==0][outcome].mean():.4f}")
    print(f"  Naive ATE (raw difference): ${df_subset[df_subset[treatment_col]==1][outcome].mean() - df_subset[df_subset[treatment_col]==0][outcome].mean():.4f}")

    X = df_subset[features].values
    T = df_subset[treatment_col].values
    Y = df_subset[outcome].values

    # CausalForestDML uses two nuisance models:
    #   model_y: predicts outcome (spend) from features
    #   model_t: predicts treatment probability from features
    # These partial out confounding before the forest estimates CATE.
    # Using GradientBoosting for both — better fit than linear for this data.
    model_y = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_STATE
    )

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=None,
        random_state=RANDOM_STATE,
        verbose=0,
        cv=5,
    )

    print(f"  Fitting causal forest (this takes a moment)...")
    cf.fit(Y, T, X=X)

    cate = cf.effect(X)
    lb, ub = cf.effect_interval(X, alpha=0.10)  # 90% confidence intervals

    print(f"  Mean CATE: ${cate.mean():.4f}")
    print(f"  Median CATE: ${np.median(cate):.4f}")
    print(f"  CATE std: ${cate.std():.4f}")
    print(f"  % customers with positive CATE: {(cate > 0).mean():.2%}")
    print()

    return cf, cate, lb, ub


# ---------------------------------------------------------------------------
# 3. Run the three comparisons
# ---------------------------------------------------------------------------

# A: Any email vs. No Email
df_any = df[df['segment'].isin(['Mens E-Mail', 'Womens E-Mail', 'No E-Mail'])].copy()
df_any['is_treated'] = (df_any['segment'] != 'No E-Mail').astype(int)
cf_a, cate_a, lb_a, ub_a = run_causal_forest(df_any, 'is_treated', FEATURES, 'spend', 'Any Email vs Control')

# B: Mens Email vs. No Email
df_mens = df[df['segment'].isin(['Mens E-Mail', 'No E-Mail'])].copy()
df_mens['is_treated'] = (df_mens['segment'] == 'Mens E-Mail').astype(int)
cf_b, cate_b, lb_b, ub_b = run_causal_forest(df_mens, 'is_treated', FEATURES, 'spend', 'Mens Email vs Control')

# C: Womens Email vs. No Email
df_womens = df[df['segment'].isin(['Womens E-Mail', 'No E-Mail'])].copy()
df_womens['is_treated'] = (df_womens['segment'] == 'Womens E-Mail').astype(int)
cf_c, cate_c, lb_c, ub_c = run_causal_forest(df_womens, 'is_treated', FEATURES, 'spend', 'Womens Email vs Control')

# ---------------------------------------------------------------------------
# 4. Attach CATE scores back to full dataframe
# ---------------------------------------------------------------------------
# Each model scored on its own subset; attach by index
df['cate_any']    = np.nan
df['cate_mens']   = np.nan
df['cate_womens'] = np.nan
df['cate_any_lb'] = np.nan
df['cate_any_ub'] = np.nan

df.loc[df_any.index, 'cate_any']    = cate_a
df.loc[df_any.index, 'cate_any_lb'] = lb_a
df.loc[df_any.index, 'cate_any_ub'] = ub_a
df.loc[df_mens.index, 'cate_mens']  = cate_b
df.loc[df_womens.index, 'cate_womens'] = cate_c

# ---------------------------------------------------------------------------
# 5. CATE distribution plots
# ---------------------------------------------------------------------------
print("Building charts...")

fig_dist = make_subplots(
    rows=1, cols=3,
    subplot_titles=['Any Email CATE ($)', 'Mens Email CATE ($)', 'Womens Email CATE ($)']
)
for i, (col, color, label) in enumerate([
    ('cate_any',    'steelblue',  'Any'),
    ('cate_mens',   '#EF553B',    'Mens'),
    ('cate_womens', '#00CC96',    'Womens'),
], 1):
    vals = df[col].dropna()
    fig_dist.add_trace(
        go.Histogram(x=vals, nbinsx=60, name=label, showlegend=False,
                     marker_color=color),
        row=1, col=i
    )
    fig_dist.add_vline(x=0, line_dash='dash', line_color='black', row=1, col=i)
    fig_dist.add_vline(x=vals.mean(), line_dash='dot', line_color='red', row=1, col=i)

fig_dist.update_layout(
    title='CATE Distribution: Estimated Incremental Spend per Customer<br><sup>Dashed=zero, Dotted=mean</sup>',
    template='plotly_white',
    height=450,
)
fig_dist.write_html('outputs/03_cate_distributions.html')
print("  Saved: outputs/03_cate_distributions.html")

# ---------------------------------------------------------------------------
# 6. CATE by quartile — does the model discriminate?
# ---------------------------------------------------------------------------
df_any['cate'] = cate_a
df_any['cate_quartile'] = pd.qcut(df_any['cate'], q=4, labels=['Q1 Low', 'Q2', 'Q3', 'Q4 High'])

quartile_stats = df_any.groupby('cate_quartile', observed=True).agg(
    n=('spend', 'count'),
    treated_spend=('spend', lambda x: x[df_any.loc[x.index, 'is_treated'] == 1].mean()),
    control_spend=('spend', lambda x: x[df_any.loc[x.index, 'is_treated'] == 0].mean()),
    mean_cate=('cate', 'mean'),
).reset_index()
quartile_stats['actual_uplift'] = quartile_stats['treated_spend'] - quartile_stats['control_spend']

print("=== Mean Spend by Predicted CATE Quartile (Any Email) ===")
print(quartile_stats.to_string(index=False))
print()

fig_q = go.Figure()
fig_q.add_trace(go.Bar(
    x=quartile_stats['cate_quartile'].astype(str),
    y=quartile_stats['treated_spend'],
    name='Treated', marker_color='steelblue'
))
fig_q.add_trace(go.Bar(
    x=quartile_stats['cate_quartile'].astype(str),
    y=quartile_stats['control_spend'],
    name='Control', marker_color='lightgray'
))
fig_q.add_trace(go.Scatter(
    x=quartile_stats['cate_quartile'].astype(str),
    y=quartile_stats['actual_uplift'],
    name='Actual Uplift ($)', mode='lines+markers',
    marker=dict(size=10, color='red'),
    yaxis='y2'
))
fig_q.update_layout(
    title='Mean Spend by Predicted CATE Quartile (Any Email vs Control)',
    xaxis_title='Predicted CATE Quartile',
    yaxis_title='Mean Spend ($)',
    yaxis2=dict(title='Actual Uplift ($)', overlaying='y', side='right', showgrid=False),
    barmode='group',
    template='plotly_white',
    height=500,
    legend=dict(x=0.01, y=0.99),
)
fig_q.write_html('outputs/03_cate_by_quartile.html')
print("  Saved: outputs/03_cate_by_quartile.html")

# ---------------------------------------------------------------------------
# 7. CATE uncertainty — confidence interval width
# ---------------------------------------------------------------------------
df_any['ci_width'] = ub_a - lb_a

fig_ci = go.Figure()
fig_ci.add_trace(go.Histogram(
    x=df_any['ci_width'], nbinsx=50,
    marker_color='steelblue', name='CI Width'
))
fig_ci.update_layout(
    title='90% Confidence Interval Width for CATE Estimates (Any Email)',
    xaxis_title='CI Width ($)',
    yaxis_title='Count',
    template='plotly_white',
    height=400,
)
fig_ci.write_html('outputs/03_cate_uncertainty.html')
print("  Saved: outputs/03_cate_uncertainty.html")

# ---------------------------------------------------------------------------
# 8. Feature importance from causal forest
# ---------------------------------------------------------------------------
feat_imp = cf_a.feature_importances_
fi_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': feat_imp
}).sort_values('importance', ascending=True)

fig_fi = go.Figure(go.Bar(
    x=fi_df['importance'], y=fi_df['feature'],
    orientation='h', marker_color='steelblue'
))
fig_fi.update_layout(
    title='Causal Forest Feature Importances (Any Email, spend outcome)',
    xaxis_title='Importance',
    template='plotly_white',
    height=450,
)
fig_fi.write_html('outputs/03_feature_importance.html')
print("  Saved: outputs/03_feature_importance.html")

# ---------------------------------------------------------------------------
# 9. Save scores CSV
# ---------------------------------------------------------------------------
scores = df[['recency', 'history', 'mens', 'womens', 'newbie',
              'zip_code', 'channel', 'segment', 'conversion', 'spend',
              'cate_any', 'cate_any_lb', 'cate_any_ub',
              'cate_mens', 'cate_womens']].copy()
scores.to_csv('outputs/scores_revenue.csv', index=False)
print("  Saved: outputs/scores_revenue.csv")

# ---------------------------------------------------------------------------
# 10. Save models and metadata
# ---------------------------------------------------------------------------
for name, model in [('cf_any', cf_a), ('cf_mens', cf_b), ('cf_womens', cf_c)]:
    with open(f'models/03_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# ATE summary
ate_a   = float(np.ravel(cf_a.ate_)[0])
ate_a_se= float(np.ravel(cf_a.ate_stderr_)[0])
ate_b   = float(np.ravel(cf_b.ate_)[0])
ate_b_se= float(np.ravel(cf_b.ate_stderr_)[0])
ate_c   = float(np.ravel(cf_c.ate_)[0])
ate_c_se= float(np.ravel(cf_c.ate_stderr_)[0])

print()
print("=== Average Treatment Effects (ATE) ===")
print(f"  Any Email:    ATE=${ate_a:.4f} +/- {ate_a_se:.4f}")
print(f"  Mens Email:   ATE=${ate_b:.4f} +/- {ate_b_se:.4f}")
print(f"  Womens Email: ATE=${ate_c:.4f} +/- {ate_c_se:.4f}")

metadata = {
    'stage': 3,
    'model': 'CausalForestDML (econml)',
    'target': 'spend',
    'features': FEATURES,
    'nuisance_model_y': 'GradientBoostingRegressor (n=100, depth=3)',
    'nuisance_model_t': 'GradientBoostingClassifier (n=100, depth=3)',
    'n_estimators': 500,
    'min_samples_leaf': 10,
    'cv_folds': 5,
    'random_state': RANDOM_STATE,
    'ate_any': round(ate_a, 4),
    'ate_any_se': round(ate_a_se, 4),
    'ate_mens': round(ate_b, 4),
    'ate_mens_se': round(ate_b_se, 4),
    'ate_womens': round(ate_c, 4),
    'ate_womens_se': round(ate_c_se, 4),
    'notes': (
        'CausalForestDML uses cross-fitting (cv=5) to partial out nuisance functions '
        'before estimating heterogeneous treatment effects. '
        'Outcome is raw spend (zero-inflated); forest handles this implicitly. '
        'ATE = average across all customers. CATE = individual estimates. '
        'CI computed at 90% level. '
        'Stage 4 will combine Stage 2 conversion uplift with Stage 3 CATE '
        'to produce expected incremental revenue per customer.'
    )
}
with open('models/03_revenue_uplift_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("  Saved: models/03_revenue_uplift_metadata.json")

print()
print("=== Stage 3 complete ===")
