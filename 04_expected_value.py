# -*- coding: utf-8 -*-
"""
Stage 4 - Joint Expected Value Model
Combines Stage 2 (conversion uplift) and Stage 3 (revenue CATE) to rank
customers by expected incremental revenue from each email type.

Decomposition:
  E[spend | email]   = P(convert | email)   * E[spend | convert, email]
  E[spend | control] = P(convert | control] * E[spend | convert, control]
  Incremental EV     = E[spend | email] - E[spend | control]

This lets us see *why* a customer ranks high:
  - conversion margin: email moves them from non-buyer to buyer
  - spend margin: when they do buy, the email causes them to spend more

Stage 3 CATE already estimates the combined EV directly.
This stage adds the decomposition, segment profiling, and send list.

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 04_expected_value.py
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import pickle

DATA_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
RANDOM_STATE = 42

FEATURES = [
    'recency', 'log_history', 'mens', 'womens', 'both_catalogs', 'newbie',
    'history_segment_enc', 'zip_code_enc', 'channel_enc',
]

# ---------------------------------------------------------------------------
# 1. Load data and scores
# ---------------------------------------------------------------------------
print("Loading data and scores...")
df = pd.read_csv(DATA_FILE)
df['log_history'] = np.log1p(df['history'])
df['both_catalogs'] = ((df['mens'] == 1) & (df['womens'] == 1)).astype(int)

le = LabelEncoder()
for col in ['history_segment', 'zip_code', 'channel']:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

scores_conv = pd.read_csv('outputs/scores_conversion.csv')
scores_rev  = pd.read_csv('outputs/scores_revenue.csv')

df['uplift_conv_any']    = scores_conv['uplift_any'].values
df['p_treat_any']        = scores_conv['p_treat_any'].values
df['p_ctrl_any']         = scores_conv['p_ctrl_any'].values
df['uplift_conv_mens']   = scores_conv['uplift_mens'].values
df['uplift_conv_womens'] = scores_conv['uplift_womens'].values
df['cate_any']           = scores_rev['cate_any'].values
df['cate_mens']          = scores_rev['cate_mens'].values
df['cate_womens']        = scores_rev['cate_womens'].values

print(f"  Loaded {len(df)} rows with uplift and CATE scores.")
print()

# ---------------------------------------------------------------------------
# 2. Spend-given-conversion models (E[spend | convert, treatment])
# Fit on converters only in each group — models what a buyer spends
# ---------------------------------------------------------------------------
print("Fitting spend-given-conversion models...")

def fit_spend_model(df_subset, features, label):
    converters = df_subset[df_subset['spend'] > 0]
    print(f"  {label}: {len(converters)} converters / {len(df_subset)} total")
    if len(converters) < 20:
        print(f"  WARNING: too few converters for reliable model in {label}")
    X = converters[features].values
    y = converters['spend'].values
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model

treated_mask = df['segment'].isin(['Mens E-Mail', 'Womens E-Mail'])
control_mask = df['segment'] == 'No E-Mail'
mens_mask    = df['segment'] == 'Mens E-Mail'
womens_mask  = df['segment'] == 'Womens E-Mail'

spend_model_treated = fit_spend_model(df[treated_mask], FEATURES, 'Any treated')
spend_model_control = fit_spend_model(df[control_mask], FEATURES, 'Control')
spend_model_mens    = fit_spend_model(df[mens_mask],    FEATURES, 'Mens email')
spend_model_womens  = fit_spend_model(df[womens_mask],  FEATURES, 'Womens email')

X_all = df[FEATURES].values

# E[spend | convert] predictions for every customer under each scenario
df['e_spend_given_conv_treat'] = spend_model_treated.predict(X_all)
df['e_spend_given_conv_ctrl']  = spend_model_control.predict(X_all)
df['e_spend_given_conv_mens']  = spend_model_mens.predict(X_all)
df['e_spend_given_conv_womens']= spend_model_womens.predict(X_all)

print()

# ---------------------------------------------------------------------------
# 3. Compute decomposed expected value
# EV = P(conv|treat)*E[spend|conv,treat] - P(conv|ctrl)*E[spend|conv,ctrl]
# Decompose into:
#   conversion_component = uplift_conv * E[spend|conv,ctrl]  (new buyers at baseline spend)
#   spend_component      = P(conv|treat) * (E[spend|conv,treat] - E[spend|conv,ctrl])
# ---------------------------------------------------------------------------
print("Computing expected value scores...")

# Any email
df['ev_any'] = (
    df['p_treat_any'] * df['e_spend_given_conv_treat']
    - df['p_ctrl_any'] * df['e_spend_given_conv_ctrl']
)
df['ev_conv_component'] = df['uplift_conv_any'] * df['e_spend_given_conv_ctrl']
df['ev_spend_component'] = df['p_treat_any'] * (
    df['e_spend_given_conv_treat'] - df['e_spend_given_conv_ctrl']
)

# Mens email (use mens-specific p_treat from Stage 2 conversion scores)
# Reconstruct p_treat for mens/womens using saved stage 2 models
with open('models/02_mt_mens.pkl', 'rb') as f: mt_mens = pickle.load(f)
with open('models/02_mc_mens.pkl', 'rb') as f: mc_mens = pickle.load(f)
with open('models/02_mt_womens.pkl', 'rb') as f: mt_womens = pickle.load(f)
with open('models/02_mc_womens.pkl', 'rb') as f: mc_womens = pickle.load(f)

p_treat_mens   = mt_mens.predict_proba(df[FEATURES])[:, 1]
p_ctrl_mens    = mc_mens.predict_proba(df[FEATURES])[:, 1]
p_treat_womens = mt_womens.predict_proba(df[FEATURES])[:, 1]
p_ctrl_womens  = mc_womens.predict_proba(df[FEATURES])[:, 1]

df['ev_mens'] = (
    p_treat_mens * df['e_spend_given_conv_mens']
    - p_ctrl_mens * df['e_spend_given_conv_ctrl']
)
df['ev_womens'] = (
    p_treat_womens * df['e_spend_given_conv_womens']
    - p_ctrl_womens * df['e_spend_given_conv_ctrl']
)

# Best email per customer
df['best_email'] = np.where(
    df['ev_mens'] > df['ev_womens'], 'Mens E-Mail', 'Womens E-Mail'
)
df['best_ev'] = df[['ev_mens', 'ev_womens']].max(axis=1)

print(f"  Mean EV (any email):    ${df['ev_any'].mean():.4f}")
print(f"  Mean EV (mens email):   ${df['ev_mens'].mean():.4f}")
print(f"  Mean EV (womens email): ${df['ev_womens'].mean():.4f}")
print(f"  % customers: best=Mens:   {(df['best_email']=='Mens E-Mail').mean():.2%}")
print(f"  % customers: best=Womens: {(df['best_email']=='Womens E-Mail').mean():.2%}")
print()

# ---------------------------------------------------------------------------
# 4. EV distribution and decomposition plot
# ---------------------------------------------------------------------------
print("Building charts...")

fig_ev = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        'Expected Incremental Revenue Distribution (Any Email)',
        'EV Decomposition: Conversion vs Spend Component'
    ]
)
fig_ev.add_trace(
    go.Histogram(x=df['ev_any'], nbinsx=60, marker_color='steelblue', showlegend=False),
    row=1, col=1
)
fig_ev.add_vline(x=0, line_dash='dash', line_color='black', row=1, col=1)
fig_ev.add_vline(x=df['ev_any'].mean(), line_dash='dot', line_color='red', row=1, col=1)

fig_ev.add_trace(
    go.Scatter(
        x=df['ev_conv_component'], y=df['ev_spend_component'],
        mode='markers',
        marker=dict(
            size=3, opacity=0.3, color=df['ev_any'],
            colorscale='RdBu', colorbar=dict(title='EV ($)'),
            cmin=-2, cmax=2,
        ),
        showlegend=False,
    ),
    row=1, col=2
)
fig_ev.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)
fig_ev.add_vline(x=0, line_dash='dash', line_color='gray', row=1, col=2)

fig_ev.update_xaxes(title_text='Conversion Component ($)', row=1, col=2)
fig_ev.update_yaxes(title_text='Spend-per-Order Component ($)', row=1, col=2)
fig_ev.update_layout(
    title='Expected Incremental Revenue: Distribution and Decomposition<br>'
          '<sup>Scatter: each dot is a customer. Color = total EV. Quadrants show what drives value.</sup>',
    template='plotly_white', height=500,
)
fig_ev.write_html('outputs/04_ev_distribution.html')
print("  Saved: outputs/04_ev_distribution.html")

# ---------------------------------------------------------------------------
# 5. Cumulative revenue curve — how much of the value is in the top N%?
# ---------------------------------------------------------------------------
df_sorted = df.sort_values('ev_any', ascending=False).reset_index(drop=True)
n = len(df_sorted)

# Actual spend in the real experiment vs. our predicted ranking
# We don't have counterfactual ground truth for every individual,
# but we can show: if we'd sent to top K%, what % of total observed
# uplift would we have captured?
df_sorted['cum_ev'] = df_sorted['ev_any'].cumsum()
df_sorted['pct_customers'] = (df_sorted.index + 1) / n * 100
df_sorted['pct_total_ev'] = df_sorted['cum_ev'] / df_sorted['ev_any'].sum() * 100

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=df_sorted['pct_customers'],
    y=df_sorted['pct_total_ev'],
    mode='lines', name='Model Ranking',
    line=dict(color='steelblue', width=2),
))
fig_cum.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100],
    mode='lines', name='Random Targeting',
    line=dict(color='gray', dash='dash'),
))
# Annotate key points
for pct in [10, 20, 50]:
    idx = int(pct / 100 * n) - 1
    captured = df_sorted.loc[idx, 'pct_total_ev']
    fig_cum.add_annotation(
        x=pct, y=captured,
        text=f"Top {pct}%<br>captures {captured:.0f}% of EV",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
        font=dict(size=10),
    )

fig_cum.update_layout(
    title='Cumulative EV Curve: How Concentrated Is the Value?<br>'
          '<sup>If targeting is perfect, curve bends sharply left = top customers carry most value</sup>',
    xaxis_title='% Customers Targeted (ranked by predicted EV)',
    yaxis_title='% of Total Expected Value Captured',
    template='plotly_white', height=500,
    legend=dict(x=0.6, y=0.1),
)
fig_cum.write_html('outputs/04_cumulative_ev.html')
print("  Saved: outputs/04_cumulative_ev.html")

# ---------------------------------------------------------------------------
# 6. Mens vs Womens email EV comparison per customer
# ---------------------------------------------------------------------------
fig_mv = go.Figure()
fig_mv.add_trace(go.Histogram2dContour(
    x=df['ev_mens'], y=df['ev_womens'],
    colorscale='Blues', showscale=False, ncontours=20,
))
fig_mv.add_trace(go.Scatter(
    x=[-3, 3], y=[-3, 3],
    mode='lines', name='Equal EV',
    line=dict(color='red', dash='dash'),
))
fig_mv.update_layout(
    title='Mens vs Womens Email EV per Customer<br>'
          '<sup>Above diagonal = Womens email better; Below = Mens email better</sup>',
    xaxis_title='EV from Mens Email ($)',
    yaxis_title='EV from Womens Email ($)',
    template='plotly_white', height=500,
)
fig_mv.write_html('outputs/04_mens_vs_womens_ev.html')
print("  Saved: outputs/04_mens_vs_womens_ev.html")

# ---------------------------------------------------------------------------
# 7. Segment profiling — who are the high-EV customers?
# ---------------------------------------------------------------------------
df['ev_quartile'] = pd.qcut(df['ev_any'], q=4,
                              labels=['Q1 Avoid', 'Q2 Marginal', 'Q3 Good', 'Q4 Target'])

profile = df.groupby('ev_quartile', observed=True).agg(
    n=('ev_any', 'count'),
    mean_ev=('ev_any', 'mean'),
    mean_history=('history', 'mean'),
    mean_recency=('recency', 'mean'),
    pct_mens=('mens', 'mean'),
    pct_womens=('womens', 'mean'),
    pct_newbie=('newbie', 'mean'),
    pct_both=('both_catalogs', 'mean'),
    actual_conv=('conversion', 'mean'),
    actual_spend=('spend', 'mean'),
).reset_index()

print("=== Customer Profile by EV Quartile ===")
print(profile.to_string(index=False))
print()

fig_prof = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        'Mean Prior Spend ($)', 'Mean Recency (months)',
        'Actual Conversion Rate (%)', 'Actual Post-Campaign Spend ($)',
        '% Mens Catalog Buyers', '% Newbies'
    ]
)
quarters = profile['ev_quartile'].astype(str).tolist()
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

for col_i, (metric, row_i, col_j) in enumerate([
    ('mean_history', 1, 1), ('mean_recency', 1, 2),
    ('actual_conv',  1, 3), ('actual_spend',  2, 1),
    ('pct_mens',     2, 2), ('pct_newbie',    2, 3),
]):
    vals = profile[metric]
    if metric in ('actual_conv', 'pct_mens', 'pct_newbie'):
        vals = vals * 100
    for i, (q, v) in enumerate(zip(quarters, vals)):
        fig_prof.add_trace(
            go.Bar(x=[q], y=[v], marker_color=colors[i], showlegend=(col_i==0),
                   name=q if col_i == 0 else None),
            row=row_i, col=col_j,
        )

fig_prof.update_layout(
    title='Customer Profile by EV Quartile — Who Are We Targeting?',
    template='plotly_white', height=700, barmode='group',
    showlegend=True,
)
fig_prof.write_html('outputs/04_segment_profiles.html')
print("  Saved: outputs/04_segment_profiles.html")

# ---------------------------------------------------------------------------
# 8. Send list — rank all customers, assign recommended email
# ---------------------------------------------------------------------------
send_list = df[[
    'recency', 'history', 'mens', 'womens', 'newbie',
    'zip_code', 'channel', 'segment', 'conversion', 'spend',
    'ev_any', 'ev_mens', 'ev_womens', 'best_email', 'best_ev',
    'uplift_conv_any', 'cate_any',
    'ev_conv_component', 'ev_spend_component',
]].copy()

send_list['ev_rank'] = send_list['ev_any'].rank(ascending=False).astype(int)
send_list = send_list.sort_values('ev_rank')

# Recommended action
send_list['recommendation'] = np.where(
    send_list['ev_any'] > 0, send_list['best_email'], 'Do Not Email'
)

rec_counts = send_list['recommendation'].value_counts()
print("=== Send Recommendations ===")
for rec, cnt in rec_counts.items():
    print(f"  {rec}: {cnt:,} customers ({cnt/len(send_list):.1%})")
print()

send_list.to_csv('outputs/send_list.csv', index=False)
print("  Saved: outputs/send_list.csv")

# ---------------------------------------------------------------------------
# 9. ROI simulation — if we email top N% by EV rank, what do we capture?
# ---------------------------------------------------------------------------
# Use actual observed spend as the outcome to validate ranking
# (Not perfect — we don't observe counterfactual — but gives directional check)
thresholds = [5, 10, 20, 30, 50, 100]
print("=== ROI Simulation: Actual Spend Captured by Top N% Targeting ===")
print(f"  {'Top %':>6}  {'N Customers':>12}  {'Actual Spend':>14}  {'% of Total Spend':>16}")
total_spend = df['spend'].sum()
for pct in thresholds:
    top_n = int(len(send_list) * pct / 100)
    top_idx = send_list.head(top_n).index
    captured = df.loc[top_idx, 'spend'].sum()
    print(f"  {pct:>5}%  {top_n:>12,}  ${captured:>13.2f}  {captured/total_spend:>15.1%}")
print()

# ---------------------------------------------------------------------------
# 10. Save metadata
# ---------------------------------------------------------------------------
metadata = {
    'stage': 4,
    'inputs': {
        'conversion_scores': 'outputs/scores_conversion.csv',
        'revenue_scores': 'outputs/scores_revenue.csv',
        'stage2_models': 'models/02_m[t|c]_[any|mens|womens].pkl',
    },
    'outputs': {
        'send_list': 'outputs/send_list.csv',
        'charts': [
            'outputs/04_ev_distribution.html',
            'outputs/04_cumulative_ev.html',
            'outputs/04_mens_vs_womens_ev.html',
            'outputs/04_segment_profiles.html',
        ]
    },
    'recommendations': {
        k: int(v) for k, v in rec_counts.items()
    },
    'mean_ev_any': round(float(df['ev_any'].mean()), 4),
    'mean_ev_mens': round(float(df['ev_mens'].mean()), 4),
    'mean_ev_womens': round(float(df['ev_womens'].mean()), 4),
    'notes': (
        'EV = P(convert|email)*E[spend|convert,email] - P(convert|ctrl)*E[spend|convert,ctrl]. '
        'Decomposed into conversion component (new buyers) and spend component (higher order value). '
        'Best email assigned per customer by comparing ev_mens vs ev_womens. '
        'Do Not Email = customers with ev_any <= 0 (negative or zero incremental value). '
        'Spend-given-conversion models trained on converters only — small sample, use with caution. '
        'Stage 3 cate_any is a more stable estimate of incremental revenue for ranking.'
    )
}
with open('models/04_expected_value_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("  Saved: models/04_expected_value_metadata.json")
print()
print("=== Stage 4 complete ===")
