# -*- coding: utf-8 -*-
"""
Stage 6 - CATE-Space Clustering
Cluster customers on their response score vectors, not their features.
Clusters are defined by HOW customers respond to email, not who they are.

After clustering, profile each cluster back against original features
to answer: what kind of customer IS each response type?

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 06_cate_clustering.py
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

DATA_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data and all response scores
# ---------------------------------------------------------------------------
print("Loading data and response scores...")
df_raw   = pd.read_csv(DATA_FILE)
df_send  = pd.read_csv('outputs/send_list.csv')
df_conv  = pd.read_csv('outputs/scores_conversion.csv')
df_rev   = pd.read_csv('outputs/scores_revenue.csv')

df = df_raw.copy()
df['both_catalogs'] = ((df['mens'] == 1) & (df['womens'] == 1)).astype(int)

# Attach all response scores
df['cate_any']           = df_rev['cate_any'].values
df['cate_mens']          = df_rev['cate_mens'].values
df['cate_womens']        = df_rev['cate_womens'].values
df['uplift_conv_any']    = df_conv['uplift_any'].values
df['uplift_conv_mens']   = df_conv['uplift_mens'].values
df['uplift_conv_womens'] = df_conv['uplift_womens'].values
df['ev_any']             = df_send['ev_any'].values
df['ev_mens']            = df_send['ev_mens'].values
df['ev_womens']          = df_send['ev_womens'].values
df['ev_conv_component']  = df_send['ev_conv_component'].values
df['ev_spend_component'] = df_send['ev_spend_component'].values

# cate_mens and cate_womens are NaN for control-only rows in those subsets
# Fill with cate_any as a reasonable fallback for clustering
df['cate_mens']   = df['cate_mens'].fillna(df['cate_any'])
df['cate_womens'] = df['cate_womens'].fillna(df['cate_any'])

RESPONSE_FEATURES = [
    'cate_any',
    'cate_mens',
    'cate_womens',
    'uplift_conv_any',
    'uplift_conv_mens',
    'uplift_conv_womens',
    'ev_conv_component',
    'ev_spend_component',
]

X_resp = df[RESPONSE_FEATURES].values
print(f"  Response feature matrix: {X_resp.shape}")
print()

# ---------------------------------------------------------------------------
# 2. Scale and choose k
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resp)

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

fig_sel = make_subplots(rows=1, cols=2,
    subplot_titles=['Elbow Plot (Inertia)', 'Silhouette Score'])
fig_sel.add_trace(go.Scatter(x=list(ks), y=inertias, mode='lines+markers',
    marker_color='steelblue'), row=1, col=1)
fig_sel.add_trace(go.Scatter(x=list(ks), y=silhouettes, mode='lines+markers',
    marker_color='darkorange'), row=1, col=2)
fig_sel.update_xaxes(title_text='k', dtick=1)
fig_sel.update_layout(title='CATE-Space KMeans: Choosing k',
    template='plotly_white', height=400, showlegend=False)
fig_sel.write_html('outputs/06_kmeans_selection.html')
print("  Saved: outputs/06_kmeans_selection.html")

best_k = int(np.argmax(silhouettes)) + 2
print(f"\n  Best k by silhouette: {best_k}")
print()

# ---------------------------------------------------------------------------
# 3. Fit final model
# ---------------------------------------------------------------------------
print(f"Fitting final KMeans with k={best_k}...")
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
df['cate_cluster'] = km_final.fit_predict(X_scaled)

print("  Cluster sizes:")
print(df['cate_cluster'].value_counts().sort_index().to_string())
print()

# ---------------------------------------------------------------------------
# 4. Profile clusters on response scores
# ---------------------------------------------------------------------------
resp_profile = df.groupby('cate_cluster')[RESPONSE_FEATURES + ['ev_any']].mean().round(4)
print("=== Cluster Response Profiles ===")
print(resp_profile.to_string())
print()

# ---------------------------------------------------------------------------
# 5. Assign persona labels based on response profile
# ---------------------------------------------------------------------------
def assign_cate_persona(row, global_means):
    cate_any   = row['cate_any']
    cate_mens  = row['cate_mens']
    cate_womens= row['cate_womens']
    ev         = row['ev_any']
    conv_comp  = row['ev_conv_component']
    spend_comp = row['ev_spend_component']

    if ev < 0:
        return 'Suppress — Organic Buyers'
    elif cate_mens > global_means['cate_mens'] * 1.5 and cate_womens < global_means['cate_womens']:
        return 'Mens Email — High Responders'
    elif cate_womens > global_means['cate_womens'] * 1.5 and cate_mens < global_means['cate_mens']:
        return 'Womens Email — High Responders'
    elif cate_any > global_means['cate_any'] * 1.5:
        if spend_comp > conv_comp:
            return 'High Value — Spend Driven'
        else:
            return 'High Value — Conversion Driven'
    elif cate_any > 0 and ev > 0:
        return 'Moderate Responders'
    else:
        return 'Marginal / Uncertain'

global_means = resp_profile.mean()
resp_profile['persona'] = resp_profile.apply(
    assign_cate_persona, axis=1, global_means=global_means)

print("=== CATE Cluster Personas ===")
print(resp_profile[['cate_any','cate_mens','cate_womens','ev_any','persona']].to_string())
print()

df['cate_persona'] = df['cate_cluster'].map(resp_profile['persona'])

# ---------------------------------------------------------------------------
# 6. Profile clusters back on ORIGINAL features
# ---------------------------------------------------------------------------
feat_profile = df.groupby('cate_cluster').agg(
    n=('cate_cluster','count'),
    mean_history=('history','mean'),
    mean_recency=('recency','mean'),
    pct_mens=('mens','mean'),
    pct_womens=('womens','mean'),
    pct_both=('both_catalogs','mean'),
    pct_newbie=('newbie','mean'),
    actual_conv=('conversion','mean'),
    actual_spend=('spend','mean'),
).round(4)
feat_profile['top_channel'] = df.groupby('cate_cluster')['channel'].agg(
    lambda x: x.value_counts().index[0])
feat_profile['top_zip'] = df.groupby('cate_cluster')['zip_code'].agg(
    lambda x: x.value_counts().index[0])
feat_profile['persona'] = resp_profile['persona']

print("=== CATE Clusters: Who Are They? (original features) ===")
print(feat_profile.to_string())
print()

# Also show actual experimental outcomes by cluster x treatment
print("=== Actual Conversion Rate by Cluster x Treatment ===")
ct = df.groupby(['cate_cluster','segment'])['conversion'].mean().unstack() * 100
ct['lift_mens']   = ct['Mens E-Mail'] - ct['No E-Mail']
ct['lift_womens'] = ct['Womens E-Mail'] - ct['No E-Mail']
print(ct.round(3).to_string())
print()

print("=== Actual Spend by Cluster x Treatment ===")
cs = df.groupby(['cate_cluster','segment'])['spend'].mean().unstack()
cs['lift_mens']   = cs['Mens E-Mail'] - cs['No E-Mail']
cs['lift_womens'] = cs['Womens E-Mail'] - cs['No E-Mail']
print(cs.round(4).to_string())
print()

# ---------------------------------------------------------------------------
# 7. PCA of response space — visualise clusters
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
df['resp_pca1'] = X_pca[:, 0]
df['resp_pca2'] = X_pca[:, 1]

sample = df.sample(n=min(10000, len(df)), random_state=RANDOM_STATE)
colors = px.colors.qualitative.Plotly

fig_pca = go.Figure()
for c in sorted(df['cate_cluster'].unique()):
    mask = sample['cate_cluster'] == c
    persona = resp_profile.loc[c, 'persona']
    fig_pca.add_trace(go.Scatter(
        x=sample[mask]['resp_pca1'],
        y=sample[mask]['resp_pca2'],
        mode='markers',
        marker=dict(size=4, opacity=0.45, color=colors[c % len(colors)]),
        name=f"C{c}: {persona}",
    ))
fig_pca.update_layout(
    title=f'CATE-Space Clusters in PCA (2 components, '
          f'{pca.explained_variance_ratio_.sum():.1%} variance explained)',
    xaxis_title='PC1 (response magnitude)',
    yaxis_title='PC2 (email type preference)',
    template='plotly_white', height=550,
)
fig_pca.write_html('outputs/06_cate_clusters_pca.html')
print("  Saved: outputs/06_cate_clusters_pca.html")

# ---------------------------------------------------------------------------
# 8. Cluster profiles: response scores heatmap
# ---------------------------------------------------------------------------
hm_resp = resp_profile[RESPONSE_FEATURES]
hm_norm = (hm_resp - hm_resp.min()) / (hm_resp.max() - hm_resp.min() + 1e-9)
row_labels = [f"C{c}: {resp_profile.loc[c,'persona']}" for c in hm_norm.index]

fig_hm_resp = go.Figure(go.Heatmap(
    z=hm_norm.values,
    x=RESPONSE_FEATURES,
    y=row_labels,
    colorscale='RdBu', zmid=0.5,
    text=hm_resp.round(3).values,
    texttemplate='%{text}',
))
fig_hm_resp.update_layout(
    title='CATE Cluster Response Profile Heatmap (normalized per column)',
    xaxis_tickangle=-30,
    template='plotly_white',
    height=max(350, best_k * 70),
)
fig_hm_resp.write_html('outputs/06_response_heatmap.html')
print("  Saved: outputs/06_response_heatmap.html")

# ---------------------------------------------------------------------------
# 9. Cluster profiles: original feature heatmap
# ---------------------------------------------------------------------------
hm_feat = feat_profile[['mean_history','mean_recency','pct_mens','pct_womens',
                          'pct_both','pct_newbie','actual_conv','actual_spend']]
hm_feat_norm = (hm_feat - hm_feat.min()) / (hm_feat.max() - hm_feat.min() + 1e-9)

fig_hm_feat = go.Figure(go.Heatmap(
    z=hm_feat_norm.values,
    x=hm_feat.columns.tolist(),
    y=row_labels,
    colorscale='Blues',
    text=hm_feat.round(3).values,
    texttemplate='%{text}',
))
fig_hm_feat.update_layout(
    title='CATE Clusters: Original Feature Profiles (who is each response type?)',
    xaxis_tickangle=-30,
    template='plotly_white',
    height=max(350, best_k * 70),
)
fig_hm_feat.write_html('outputs/06_feature_heatmap.html')
print("  Saved: outputs/06_feature_heatmap.html")

# ---------------------------------------------------------------------------
# 10. Mens vs Womens CATE scatter coloured by cluster
# ---------------------------------------------------------------------------
sample2 = df.sample(n=min(10000, len(df)), random_state=RANDOM_STATE)

fig_mv = go.Figure()
for c in sorted(df['cate_cluster'].unique()):
    mask = sample2['cate_cluster'] == c
    persona = resp_profile.loc[c, 'persona']
    fig_mv.add_trace(go.Scatter(
        x=sample2[mask]['cate_mens'],
        y=sample2[mask]['cate_womens'],
        mode='markers',
        marker=dict(size=3, opacity=0.35, color=colors[c % len(colors)]),
        name=f"C{c}: {persona}",
    ))
fig_mv.add_trace(go.Scatter(
    x=[-5, 10], y=[-5, 10], mode='lines',
    line=dict(color='gray', dash='dash'), name='Equal response', showlegend=True,
))
fig_mv.add_hline(y=0, line_color='lightgray', line_dash='dot')
fig_mv.add_vline(x=0, line_color='lightgray', line_dash='dot')
fig_mv.update_layout(
    title='Mens vs Womens CATE by Cluster<br>'
          '<sup>Above diagonal = Womens email more effective; Below = Mens email more effective</sup>',
    xaxis_title='CATE — Mens Email ($)',
    yaxis_title='CATE — Womens Email ($)',
    template='plotly_white', height=560,
)
fig_mv.write_html('outputs/06_mens_vs_womens_cate.html')
print("  Saved: outputs/06_mens_vs_womens_cate.html")

# ---------------------------------------------------------------------------
# 11. Bar chart: actual lift per cluster
# ---------------------------------------------------------------------------
lift_data = []
for c in sorted(df['cate_cluster'].unique()):
    mask = df['cate_cluster'] == c
    ctrl  = df[mask & (df['segment'] == 'No E-Mail')]
    mens  = df[mask & (df['segment'] == 'Mens E-Mail')]
    womens= df[mask & (df['segment'] == 'Womens E-Mail')]
    persona = resp_profile.loc[c, 'persona']
    lift_data.append({
        'label': f"C{c}",
        'persona': persona,
        'mens_spend_lift':   mens['spend'].mean()   - ctrl['spend'].mean(),
        'womens_spend_lift': womens['spend'].mean() - ctrl['spend'].mean(),
        'n': mask.sum(),
    })
ld = pd.DataFrame(lift_data)

fig_lift = go.Figure()
fig_lift.add_trace(go.Bar(
    x=ld['label'], y=ld['mens_spend_lift'],
    name='Mens Email Spend Lift', marker_color='steelblue',
))
fig_lift.add_trace(go.Bar(
    x=ld['label'], y=ld['womens_spend_lift'],
    name='Womens Email Spend Lift', marker_color='#EF553B',
))
fig_lift.add_hline(y=0, line_color='black', line_dash='dash')
fig_lift.update_layout(
    title='Actual Spend Lift by CATE Cluster<br>'
          '<sup>Actual observed experimental outcomes — validates the cluster response profiles</sup>',
    xaxis_title='Cluster',
    yaxis_title='Spend Lift vs Control ($)',
    barmode='group',
    template='plotly_white', height=480,
    xaxis=dict(ticktext=[f"C{r['label'][-1]}<br>{r['persona']}" for _, r in ld.iterrows()],
               tickvals=ld['label'].tolist()),
)
fig_lift.write_html('outputs/06_cluster_lift_bars.html')
print("  Saved: outputs/06_cluster_lift_bars.html")

# ---------------------------------------------------------------------------
# 12. Save outputs
# ---------------------------------------------------------------------------
df[['cate_cluster','cate_persona']].to_csv('outputs/cluster_assignments_cate.csv', index=False)
print("  Saved: outputs/cluster_assignments_cate.csv")

combined_profile = feat_profile.copy()
combined_profile[RESPONSE_FEATURES] = resp_profile[RESPONSE_FEATURES]
combined_profile.to_csv('outputs/cate_cluster_profiles.csv')
print("  Saved: outputs/cate_cluster_profiles.csv")

with open('models/06_cate_cluster_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/06_cate_cluster_kmeans.pkl', 'wb') as f:
    pickle.dump(km_final, f)

metadata = {
    'stage': 6,
    'model': 'KMeans on CATE/uplift scores',
    'k': best_k,
    'response_features': RESPONSE_FEATURES,
    'scaling': 'StandardScaler',
    'random_state': RANDOM_STATE,
    'silhouette_score': float(round(silhouettes[best_k - 2], 4)),
    'personas': {
        int(c): resp_profile.loc[c, 'persona']
        for c in resp_profile.index
    },
    'notes': (
        'Clusters defined by response score vectors, not customer features. '
        'Profiles mapped back to original features to explain who each type is. '
        'Actual experimental lift per cluster validates the clustering. '
        'Use these cluster assignments as named segments in reporting.'
    )
}
with open('models/06_cate_clustering_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("  Saved: models/06_cate_clustering_metadata.json")
print()
print("=== Stage 6 complete ===")
