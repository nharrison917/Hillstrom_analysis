# -*- coding: utf-8 -*-
"""
Stage 5 - Feature-Space Clustering Mapped to Lift Space
Cluster customers on their attributes, then ask: do natural groups
separate when viewed through the lens of treatment responsiveness?

This is a validation/communication exercise, not a targeting model.
It answers: "do our customer personas respond differently to email?"

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 05_feature_clustering.py
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pickle

from preprocessing import prepare_features, FEATURES, RANDOM_STATE, DATA_FILE

CLUSTER_FEATURES = FEATURES  # cluster on the same feature set used in models

# ---------------------------------------------------------------------------
# 1. Load and prepare
# ---------------------------------------------------------------------------
print("Loading data...")
df = prepare_features(pd.read_csv(DATA_FILE))
df_scores = pd.read_csv('outputs/send_list.csv')

# Attach scores
df['ev_any']          = df_scores['ev_any'].values
df['cate_any']        = df_scores['cate_any'].values
df['uplift_conv_any'] = df_scores['uplift_conv_any'].values
df['recommendation']  = df_scores['recommendation'].values

# Treatment group lift columns (actual observed, not modeled)
df['is_treated'] = (df['segment'] != 'No E-Mail').astype(int)

X_cluster = df[CLUSTER_FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print(f"  Feature matrix for clustering: {X_scaled.shape}")
print()

# ---------------------------------------------------------------------------
# 2. Choose k: elbow + silhouette
# ---------------------------------------------------------------------------
print("Finding optimal k (2-10)...")
inertias    = []
silhouettes = []
ks = range(2, 11)

for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=10000, random_state=RANDOM_STATE)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

fig_elbow = make_subplots(rows=1, cols=2,
    subplot_titles=['Elbow Plot (Inertia)', 'Silhouette Score'])
fig_elbow.add_trace(go.Scatter(x=list(ks), y=inertias, mode='lines+markers',
    marker_color='steelblue'), row=1, col=1)
fig_elbow.add_trace(go.Scatter(x=list(ks), y=silhouettes, mode='lines+markers',
    marker_color='darkorange'), row=1, col=2)
fig_elbow.update_xaxes(title_text='k', dtick=1)
fig_elbow.update_layout(title='K-Means: Choosing k', template='plotly_white',
    height=400, showlegend=False)
fig_elbow.write_html('outputs/05_kmeans_selection.html')
print("  Saved: outputs/05_kmeans_selection.html")

best_k = int(np.argmax(silhouettes)) + 2   # ks starts at 2
print(f"\n  Best k by silhouette: {best_k}")
print()

# ---------------------------------------------------------------------------
# 3. Fit final model at best_k
# ---------------------------------------------------------------------------
print(f"Fitting final KMeans with k={best_k}...")
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
df['cluster'] = km_final.fit_predict(X_scaled)

print("  Cluster sizes:")
print(df['cluster'].value_counts().sort_index().to_string())
print()

# ---------------------------------------------------------------------------
# 4. Profile clusters on features
# ---------------------------------------------------------------------------
profile = df.groupby('cluster').agg(
    n=('cluster', 'count'),
    mean_history=('history', 'mean'),
    mean_recency=('recency', 'mean'),
    pct_mens=('mens', 'mean'),
    pct_womens=('womens', 'mean'),
    pct_both=('both_catalogs', 'mean'),
    pct_newbie=('newbie', 'mean'),
).round(3)

# Most common channel and zip per cluster
profile['top_channel'] = df.groupby('cluster')['channel'].agg(
    lambda x: x.value_counts().index[0])
profile['top_zip'] = df.groupby('cluster')['zip_code'].agg(
    lambda x: x.value_counts().index[0])

print("=== Cluster Feature Profiles ===")
print(profile.to_string())
print()

# ---------------------------------------------------------------------------
# 5. Map clusters onto lift space (the key question)
# ---------------------------------------------------------------------------
# Compute actual conversion lift and spend lift per cluster
# using the real experimental groups within each cluster
lift_rows = []
for c in sorted(df['cluster'].unique()):
    mask = df['cluster'] == c
    t = df[mask & (df['segment'] != 'No E-Mail')]
    ctrl = df[mask & (df['segment'] == 'No E-Mail')]
    t_mens = df[mask & (df['segment'] == 'Mens E-Mail')]
    t_womens = df[mask & (df['segment'] == 'Womens E-Mail')]

    conv_lift   = t['conversion'].mean() - ctrl['conversion'].mean() if len(ctrl) > 0 else np.nan
    spend_lift  = t['spend'].mean()      - ctrl['spend'].mean()      if len(ctrl) > 0 else np.nan
    mens_lift   = t_mens['conversion'].mean() - ctrl['conversion'].mean() if len(ctrl) > 0 else np.nan
    womens_lift = t_womens['conversion'].mean() - ctrl['conversion'].mean() if len(ctrl) > 0 else np.nan
    mean_ev     = df[mask]['ev_any'].mean()
    mean_cate   = df[mask]['cate_any'].mean()

    lift_rows.append({
        'cluster': c,
        'n': mask.sum(),
        'conv_lift': conv_lift,
        'spend_lift': spend_lift,
        'mens_conv_lift': mens_lift,
        'womens_conv_lift': womens_lift,
        'mean_ev': mean_ev,
        'mean_cate': mean_cate,
        'mean_history': df[mask]['history'].mean(),
        'mean_recency': df[mask]['recency'].mean(),
        'top_channel': df[mask]['channel'].value_counts().index[0],
        'top_zip': df[mask]['zip_code'].value_counts().index[0],
        'pct_both': df[mask]['both_catalogs'].mean(),
        'pct_newbie': df[mask]['newbie'].mean(),
    })

lift_df = pd.DataFrame(lift_rows)

print("=== Clusters in Lift Space ===")
print(lift_df[['cluster','n','conv_lift','spend_lift','mean_cate',
               'top_channel','top_zip','pct_both']].to_string(index=False))
print()

# Assign descriptive persona labels based on lift profile
def assign_persona(row):
    if row['conv_lift'] < 0.001 and row['spend_lift'] < 0.1:
        return 'Organic Buyers (Suppress)'
    elif row['mean_cate'] > lift_df['mean_cate'].quantile(0.75):
        return 'High Responders'
    elif row['mean_cate'] < lift_df['mean_cate'].quantile(0.25):
        return 'Low Responders'
    elif row['mens_conv_lift'] > row['womens_conv_lift'] * 1.5:
        return 'Mens-Responsive'
    elif row['womens_conv_lift'] > row['mens_conv_lift'] * 1.5:
        return 'Womens-Responsive'
    else:
        return 'Moderate Responders'

lift_df['persona'] = lift_df.apply(assign_persona, axis=1)
df['persona'] = df['cluster'].map(lift_df.set_index('cluster')['persona'])

print("=== Cluster Personas ===")
print(lift_df[['cluster','n','persona','mean_cate','top_channel','top_zip']].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 6. Lift map: bubble chart — clusters in conversion lift x spend lift space
# ---------------------------------------------------------------------------
colors = px.colors.qualitative.Plotly
fig_lift = go.Figure()

for _, row in lift_df.iterrows():
    c = int(row['cluster'])
    fig_lift.add_trace(go.Scatter(
        x=[row['conv_lift'] * 100],
        y=[row['spend_lift']],
        mode='markers+text',
        marker=dict(
            size=np.sqrt(row['n']) * 0.8,
            color=colors[c % len(colors)],
            opacity=0.8,
            line=dict(width=1, color='white'),
        ),
        text=[f"C{c}: {row['persona']}"],
        textposition='top center',
        name=f"Cluster {c} ({row['persona']})",
        hovertemplate=(
            f"<b>Cluster {c}</b><br>"
            f"Persona: {row['persona']}<br>"
            f"n={row['n']:,}<br>"
            f"Conv lift: {row['conv_lift']*100:.3f}pp<br>"
            f"Spend lift: ${row['spend_lift']:.4f}<br>"
            f"Mean CATE: ${row['mean_cate']:.4f}<br>"
            f"Top channel: {row['top_channel']}<br>"
            f"Top zip: {row['top_zip']}<br>"
            f"% Both catalogs: {row['pct_both']:.1%}<extra></extra>"
        ),
    ))

fig_lift.add_hline(y=0, line_dash='dash', line_color='gray')
fig_lift.add_vline(x=0, line_dash='dash', line_color='gray')
fig_lift.update_layout(
    title='Feature Clusters Mapped to Lift Space<br>'
          '<sup>Bubble size = cluster size. Axes = actual observed lift in experiment.</sup>',
    xaxis_title='Conversion Lift (percentage points)',
    yaxis_title='Spend Lift ($)',
    template='plotly_white',
    height=600,
    legend=dict(x=1.01, y=1),
)
fig_lift.write_html('outputs/05_clusters_lift_map.html')
print("  Saved: outputs/05_clusters_lift_map.html")

# ---------------------------------------------------------------------------
# 7. PCA visualization — clusters in 2D feature space
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

sample = df.sample(n=min(8000, len(df)), random_state=RANDOM_STATE)
fig_pca = go.Figure()
for c in sorted(df['cluster'].unique()):
    mask = sample['cluster'] == c
    persona = lift_df.loc[lift_df['cluster']==c, 'persona'].values[0]
    fig_pca.add_trace(go.Scatter(
        x=sample[mask]['pca1'], y=sample[mask]['pca2'],
        mode='markers',
        marker=dict(size=3, opacity=0.4, color=colors[c % len(colors)]),
        name=f"C{c}: {persona}",
    ))
fig_pca.update_layout(
    title=f'Feature Clusters in PCA Space (2 components, {pca.explained_variance_ratio_.sum():.1%} variance explained)',
    xaxis_title='PC1', yaxis_title='PC2',
    template='plotly_white', height=500,
)
fig_pca.write_html('outputs/05_feature_clusters_pca.html')
print("  Saved: outputs/05_feature_clusters_pca.html")

# ---------------------------------------------------------------------------
# 8. Cluster profile heatmap
# ---------------------------------------------------------------------------
heatmap_features = ['mean_history', 'mean_recency', 'pct_both', 'pct_newbie',
                     'conv_lift', 'spend_lift', 'mean_cate', 'mean_ev']
hm_df = lift_df[['cluster'] + heatmap_features].set_index('cluster')
hm_norm = (hm_df - hm_df.min()) / (hm_df.max() - hm_df.min() + 1e-9)

fig_hm = go.Figure(go.Heatmap(
    z=hm_norm.values,
    x=heatmap_features,
    y=[f"C{c}" for c in hm_norm.index],
    colorscale='RdBu',
    zmid=0.5,
    text=hm_df.round(3).values,
    texttemplate='%{text}',
    hoverongaps=False,
))
fig_hm.update_layout(
    title='Cluster Profile Heatmap (normalized 0-1 per column)',
    xaxis_tickangle=-30,
    template='plotly_white',
    height=max(300, best_k * 60),
)
fig_hm.write_html('outputs/05_cluster_heatmap.html')
print("  Saved: outputs/05_cluster_heatmap.html")

# ---------------------------------------------------------------------------
# 9. Save cluster assignments and metadata
# ---------------------------------------------------------------------------
df[['cluster', 'persona']].to_csv('outputs/cluster_assignments_feature.csv', index=False)
lift_df.to_csv('outputs/cluster_lift_profiles.csv', index=False)
print("  Saved: outputs/cluster_assignments_feature.csv")
print("  Saved: outputs/cluster_lift_profiles.csv")

with open('models/05_feature_cluster_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/05_feature_cluster_kmeans.pkl', 'wb') as f:
    pickle.dump(km_final, f)

metadata = {
    'stage': 5,
    'model': 'KMeans',
    'k': best_k,
    'features': CLUSTER_FEATURES,
    'scaling': 'StandardScaler',
    'random_state': RANDOM_STATE,
    'silhouette_score': float(round(silhouettes[best_k - 2], 4)),
    'personas': lift_df[['cluster', 'persona', 'n', 'mean_cate']].to_dict(orient='records'),
    'notes': (
        'Feature-space clustering — clusters defined by who customers are, '
        'not how they respond. Mapped onto lift space using actual experimental '
        'outcomes to ask whether natural customer groups happen to differ in '
        'email responsiveness. This is descriptive/communication, not targeting. '
        'Use CATE-space clustering (Stage 6) for response-defined segments.'
    )
}
with open('models/05_feature_clustering_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("  Saved: models/05_feature_clustering_metadata.json")
print()
print("=== Stage 5 complete ===")
