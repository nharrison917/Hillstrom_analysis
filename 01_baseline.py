# -*- coding: utf-8 -*-
"""
Stage 1 - Baseline XGBoost Classifier
Target: binary conversion
Purpose: feature importances, data sanity check, preprocessing pipeline

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 01_baseline.py
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import pickle

DATA_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and inspect
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Conversion rate: {df['conversion'].mean():.4f} ({df['conversion'].sum()} conversions)")
print(f"  Visit rate: {df['visit'].mean():.4f}")
print(f"  Spend > 0: {(df['spend'] > 0).sum()} records")
print()

# Treatment group breakdown
print("Treatment groups:")
print(df['segment'].value_counts())
print()

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
print("Engineering features...")

df_model = df.copy()

# Log-transform history (right-skewed, $29.99 floor)
df_model['log_history'] = np.log1p(df_model['history'])

# Overlap segment flag
df_model['both_catalogs'] = ((df_model['mens'] == 1) & (df_model['womens'] == 1)).astype(int)

# Encode categoricals
cat_cols = ['history_segment', 'zip_code', 'channel']
le = LabelEncoder()
for col in cat_cols:
    df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))

# Encode treatment (for reference — not used as a feature in baseline)
segment_map = {'No E-Mail': 0, 'Mens E-Mail': 1, 'Womens E-Mail': 2}
df_model['treatment'] = df_model['segment'].map(segment_map)

# ---------------------------------------------------------------------------
# 3. Feature matrix
# ---------------------------------------------------------------------------
# Note: we exclude treatment assignment from features — this is a baseline
# "who converts" model, not an uplift model. Treatment goes in at Stage 2.
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
TARGET = 'conversion'

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"  Feature matrix: {X.shape}")
print(f"  Class balance: {y.value_counts().to_dict()}")
print()

# ---------------------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# 5. Train XGBoost
# ---------------------------------------------------------------------------
print("Training XGBoost baseline...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight (class imbalance ratio): {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    verbosity=0,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("  Done.")

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print()
print("=== Test Set Results ===")
print(f"  ROC-AUC: {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=['No Convert', 'Convert']))

# Cross-val AUC on full dataset
print("Running 5-fold cross-validation (full dataset)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"  CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print()

# ---------------------------------------------------------------------------
# 7. Feature importance plot
# ---------------------------------------------------------------------------
print("Building feature importance chart...")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': importances
}).sort_values('importance', ascending=True)

fig_imp = go.Figure(go.Bar(
    x=feat_df['importance'],
    y=feat_df['feature'],
    orientation='h',
    marker_color='steelblue',
))
fig_imp.update_layout(
    title='XGBoost Feature Importances - Baseline Conversion Model',
    xaxis_title='Importance (gain)',
    yaxis_title='Feature',
    height=500,
    template='plotly_white',
)
fig_imp.write_html('outputs/01_feature_importance.html')
print("  Saved: outputs/01_feature_importance.html")

# ---------------------------------------------------------------------------
# 8. Conversion rate by treatment group (sanity check)
# ---------------------------------------------------------------------------
print("Building treatment group comparison chart...")

treat_stats = df.groupby('segment').agg(
    n=('conversion', 'count'),
    conversions=('conversion', 'sum'),
    conversion_rate=('conversion', 'mean'),
    mean_spend=('spend', 'mean'),
    median_spend=('spend', 'median'),
).reset_index()
treat_stats['conversion_pct'] = treat_stats['conversion_rate'] * 100

print()
print("=== Treatment Group Summary ===")
print(treat_stats.to_string(index=False))

fig_treat = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Conversion Rate by Segment (%)', 'Mean Spend by Segment ($)']
)
colors = ['#636EFA', '#EF553B', '#00CC96']
for i, row in treat_stats.iterrows():
    fig_treat.add_trace(
        go.Bar(name=row['segment'], x=[row['segment']], y=[row['conversion_pct']],
               marker_color=colors[i], showlegend=False),
        row=1, col=1
    )
    fig_treat.add_trace(
        go.Bar(name=row['segment'], x=[row['segment']], y=[row['mean_spend']],
               marker_color=colors[i], showlegend=True),
        row=1, col=2
    )
fig_treat.update_layout(
    title='Baseline Outcomes by Treatment Group',
    template='plotly_white',
    height=450,
    barmode='group',
)
fig_treat.write_html('outputs/01_treatment_comparison.html')
print("  Saved: outputs/01_treatment_comparison.html")

# ---------------------------------------------------------------------------
# 9. Spend distribution (zero-inflated visualization)
# ---------------------------------------------------------------------------
print("Building spend distribution chart...")

spenders = df[df['spend'] > 0]['spend']
fig_spend = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f'Spend Distribution (converters only, n={len(spenders)})',
        'Log Spend Distribution'
    ]
)
fig_spend.add_trace(
    go.Histogram(x=spenders, nbinsx=50, marker_color='steelblue', name='spend'),
    row=1, col=1
)
fig_spend.add_trace(
    go.Histogram(x=np.log1p(spenders), nbinsx=50, marker_color='darkorange', name='log spend'),
    row=1, col=2
)
fig_spend.update_layout(
    title='Spend Distribution Among Converters',
    template='plotly_white',
    height=400,
    showlegend=False,
)
fig_spend.write_html('outputs/01_spend_distribution.html')
print("  Saved: outputs/01_spend_distribution.html")

# ---------------------------------------------------------------------------
# 10. Save model and metadata
# ---------------------------------------------------------------------------
print("Saving model...")

with open('models/01_baseline_xgb.pkl', 'wb') as f:
    pickle.dump(model, f)

metadata = {
    'stage': 1,
    'model': 'XGBClassifier',
    'target': TARGET,
    'features': FEATURES,
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'scale_pos_weight': float(round(scale_pos_weight, 4)),
    'random_state': RANDOM_STATE,
    'test_roc_auc': float(round(auc, 4)),
    'cv_roc_auc_mean': float(round(cv_scores.mean(), 4)),
    'cv_roc_auc_std': float(round(cv_scores.std(), 4)),
    'train_rows': int(X_train.shape[0]),
    'test_rows': int(X_test.shape[0]),
    'notes': (
        'Baseline only — treatment is excluded from features intentionally. '
        'This answers who converts, not who is persuaded. '
        'log_history used instead of raw history (right-skewed, $29.99 floor). '
        'both_catalogs flag added for overlap segment (6448 records).'
    )
}
with open('models/01_baseline_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("  Saved: models/01_baseline_xgb.pkl")
print("  Saved: models/01_baseline_metadata.json")
print()
print("=== Stage 1 complete ===")
