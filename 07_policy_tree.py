# -*- coding: utf-8 -*-
"""
Stage 7 - Policy Tree
Learns shallow decision rules that maximize expected incremental revenue.
Output is a small set of if/then rules deployable without a scoring pipeline.

Uses econml PolicyTree fitted on CATE estimates from Stage 3.
Three trees:
  A) Any email vs. No Email
  B) Mens Email vs. No Email
  C) Womens Email vs. No Email

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 07_policy_tree.py
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from econml.policy import PolicyTree
import pickle

from preprocessing import prepare_features, FEATURES, RANDOM_STATE, DATA_FILE

# ---------------------------------------------------------------------------
# 1. Load data and CATE scores
# ---------------------------------------------------------------------------
print("Loading data...")
df = prepare_features(pd.read_csv(DATA_FILE))
df_rev = pd.read_csv('outputs/scores_revenue.csv')

# Reverse map for history_segment (only categorical that still needs display decoding)
le_hist = LabelEncoder()
le_hist.fit(df['history_segment'].astype(str))
rev_hist = {int(le_hist.transform([c])[0]): c for c in le_hist.classes_}

df['cate_any']    = df_rev['cate_any'].values
df['cate_mens']   = df_rev['cate_mens'].fillna(df_rev['cate_any']).values \
                    if 'cate_mens' in df_rev else df_rev['cate_any'].values
df['cate_womens'] = df_rev['cate_womens'].fillna(df_rev['cate_any']).values \
                    if 'cate_womens' in df_rev else df_rev['cate_any'].values

X = df[FEATURES].values
print(f"  Loaded {len(df)} rows.")
print()

# ---------------------------------------------------------------------------
# 2. Fit policy trees
# ---------------------------------------------------------------------------
def fit_policy_tree(cate_values, X, label, max_depth=3):
    """
    PolicyTree maximises E[CATE * policy(x)] over the feature space.
    Pass [0, cate] as two columns: control baseline = 0, treatment = cate.
    predict_value returns per-leaf mean CATE; threshold at 0 for send/suppress.
    """
    print(f"--- Policy Tree: {label} ---")
    pt = PolicyTree(
        max_depth=max_depth,
        min_samples_leaf=500,
        random_state=RANDOM_STATE,
    )
    cate_2col = np.column_stack([np.zeros(len(cate_values)), cate_values])
    pt.fit(X, cate_2col)
    print(f"  Fitted. Depth: {pt.get_depth()}")
    return pt


pt_any    = fit_policy_tree(df['cate_any'].values,    X, 'Any Email',    max_depth=3)
pt_mens   = fit_policy_tree(df['cate_mens'].values,   X, 'Mens Email',   max_depth=3)
pt_womens = fit_policy_tree(df['cate_womens'].values, X, 'Womens Email', max_depth=3)
print()

# ---------------------------------------------------------------------------
# 3. Extract and display rules
# ---------------------------------------------------------------------------
def tree_to_rules(pt, feature_names, rev_hist, label):
    """
    Walk the policy tree and produce human-readable rules.
    Each leaf: show the rule path, recommended action, and mean CATE in that leaf.

    With one-hot encoding, zip and channel columns are directly readable
    (zip_Suburban = 1, channel_Multichannel = 1, etc.). Only history_segment_enc
    still needs a reverse lookup to display the original tier labels.
    """
    tree = pt.tree_
    lines = [f"\n=== Policy Tree: {label} ==="]
    lines.append(f"  Depth: {pt.get_depth()} | Leaves: {tree.n_leaves}")
    lines.append("")

    # Binary features where threshold is always ~0.5 (0 vs 1)
    _BINARY = {
        'mens', 'womens', 'both_catalogs', 'newbie',
        'zip_Suburban', 'zip_Urban',
        'channel_Multichannel', 'channel_Web',
    }

    def fmt_threshold(feat_idx, threshold):
        fname = feature_names[feat_idx]
        if fname == 'log_history':
            val = np.expm1(threshold)
            return f"history <= ${val:.0f}", f"history > ${val:.0f}"
        elif fname == 'history_segment_enc':
            below = [v for k, v in rev_hist.items() if k <= threshold]
            above = [v for k, v in rev_hist.items() if k > threshold]
            return f"history_segment in {below}", f"history_segment in {above}"
        elif fname in _BINARY:
            return f"{fname} = 0", f"{fname} = 1"
        else:
            return f"{fname} <= {threshold:.2f}", f"{fname} > {threshold:.2f}"

    def recurse(node_id, depth, path):
        indent = "  " * (depth + 1)
        feat  = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        value = tree.value[node_id]

        if feat < 0:  # leaf node
            mean_cate = float(value[1][0]) if len(value) > 1 else float(value[0][0])
            action = "SEND EMAIL" if mean_cate > 0 else "DO NOT EMAIL"
            rule = " AND ".join(path) if path else "(all customers)"
            lines.append(f"{indent}IF {rule}")
            lines.append(f"{indent}  -> {action}  (mean CATE: ${mean_cate:.4f})")
            lines.append("")
            return

        left_desc, right_desc = fmt_threshold(feat, thresh)
        recurse(tree.children_left[node_id],  depth+1, path + [left_desc])
        recurse(tree.children_right[node_id], depth+1, path + [right_desc])

    recurse(0, 0, [])
    return "\n".join(lines)


rules_any    = tree_to_rules(pt_any,    FEATURES, rev_hist, 'Any Email vs No Email')
rules_mens   = tree_to_rules(pt_mens,   FEATURES, rev_hist, 'Mens Email vs No Email')
rules_womens = tree_to_rules(pt_womens, FEATURES, rev_hist, 'Womens Email vs No Email')

print(rules_any)
print(rules_mens)
print(rules_womens)

with open('outputs/07_policy_rules.txt', 'w', encoding='utf-8') as f:
    f.write(rules_any + "\n")
    f.write(rules_mens + "\n")
    f.write(rules_womens + "\n")
print("Saved: outputs/07_policy_rules.txt")
print()

# ---------------------------------------------------------------------------
# 4. Apply trees and compute leaf-level statistics
# ---------------------------------------------------------------------------
def leaf_stats(pt, cate_col, X, df, label):
    """
    Assign each customer to a leaf, compute actual lift and mean CATE per leaf.
    predict_value[:,1] is the treatment column (email) per-leaf mean CATE.
    Action = send if leaf CATE > 0.
    """
    leaf_ids    = pt.apply(X)
    leaf_values = pt.predict_value(X)[:, 1]  # treatment column
    df = df.copy()
    df['leaf']          = leaf_ids
    df['leaf_cate']     = leaf_values
    df['policy_action'] = (leaf_values > 0).astype(int)

    stats = []
    for leaf in sorted(df['leaf'].unique()):
        mask  = df['leaf'] == leaf
        treat = df[mask & (df['segment'] != 'No E-Mail')]
        ctrl  = df[mask & (df['segment'] == 'No E-Mail')]
        leaf_cate = df[mask]['leaf_cate'].iloc[0]
        stats.append({
            'leaf': leaf,
            'n': mask.sum(),
            'action': 'Send' if leaf_cate > 0 else 'No Send',
            'mean_cate': leaf_cate,
            'actual_conv_treated': treat['conversion'].mean() if len(treat) > 0 else np.nan,
            'actual_conv_control': ctrl['conversion'].mean()  if len(ctrl)  > 0 else np.nan,
            'actual_spend_lift':   treat['spend'].mean() - ctrl['spend'].mean()
                                   if (len(treat) > 0 and len(ctrl) > 0) else np.nan,
        })

    sdf = pd.DataFrame(stats).sort_values('mean_cate', ascending=False)
    print(f"=== Leaf Stats: {label} ===")
    print(sdf.to_string(index=False))
    print()
    return sdf, df['policy_action']

ls_any,    policy_any    = leaf_stats(pt_any,    'cate_any',    X, df, 'Any Email')
ls_mens,   policy_mens   = leaf_stats(pt_mens,   'cate_mens',   X, df, 'Mens Email')
ls_womens, policy_womens = leaf_stats(pt_womens, 'cate_womens', X, df, 'Womens Email')

# ---------------------------------------------------------------------------
# 5. Policy comparison: tree vs. blanket send vs. no send
# ---------------------------------------------------------------------------
print("=== Policy Value Comparison (Any Email) ===")
cate = df['cate_any'].values
ev_blanket   = cate.mean()
ev_no_send   = 0.0
send_mask    = policy_any == 1
ev_tree      = cate[send_mask].mean() if send_mask.sum() > 0 else 0.0
n_send_tree  = send_mask.sum()
pct_sent     = n_send_tree / len(df) * 100

print(f"  Blanket send (everyone): mean CATE = ${ev_blanket:.4f}")
print(f"  No send:                 mean CATE = $0.0000")
print(f"  Policy tree:             mean CATE = ${ev_tree:.4f}  (sends to {n_send_tree:,} = {pct_sent:.1f}% of customers)")
print()

# ---------------------------------------------------------------------------
# 6. Visualisation: leaf value chart
# ---------------------------------------------------------------------------
def leaf_bar_chart(ls, label, color_send, color_nosend):
    fig = go.Figure()
    for _, row in ls.iterrows():
        color = color_send if row['action'] == 'Send' else color_nosend
        fig.add_trace(go.Bar(
            x=[f"Leaf {int(row['leaf'])} (n={int(row['n']):,})"],
            y=[row['mean_cate']],
            marker_color=color,
            name=row['action'],
            showlegend=False,
            hovertemplate=(
                f"Leaf {int(row['leaf'])}<br>"
                f"n={int(row['n']):,}<br>"
                f"Action: {row['action']}<br>"
                f"Mean CATE: ${row['mean_cate']:.4f}<br>"
                f"Actual spend lift: ${row['actual_spend_lift']:.4f}<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    # Legend
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color_send,   name='Send'))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color_nosend, name='No Send'))
    fig.update_layout(
        title=f'Policy Tree Leaf Values: {label}<br>'
              '<sup>Green = tree recommends sending. Red = suppress. Height = mean CATE ($).</sup>',
        xaxis_title='Leaf (sorted by CATE)',
        yaxis_title='Mean CATE ($)',
        template='plotly_white', height=480,
    )
    return fig

fig_any = leaf_bar_chart(
    ls_any.sort_values('mean_cate', ascending=False),
    'Any Email', 'steelblue', '#d62728'
)
fig_any.write_html('outputs/07_policy_tree_any.html')
print("  Saved: outputs/07_policy_tree_any.html")

fig_mens = leaf_bar_chart(
    ls_mens.sort_values('mean_cate', ascending=False),
    'Mens Email', '#2ca02c', '#d62728'
)
fig_mens.write_html('outputs/07_policy_tree_mens.html')
print("  Saved: outputs/07_policy_tree_mens.html")

fig_womens = leaf_bar_chart(
    ls_womens.sort_values('mean_cate', ascending=False),
    'Womens Email', '#9467bd', '#d62728'
)
fig_womens.write_html('outputs/07_policy_tree_womens.html')
print("  Saved: outputs/07_policy_tree_womens.html")

# ---------------------------------------------------------------------------
# 7. Combined policy: per customer, which action does the full tree recommend?
# ---------------------------------------------------------------------------
df['pt_any']    = policy_any
df['pt_mens']   = policy_mens
df['pt_womens'] = policy_womens

# Final combined recommendation: if both trees say send, pick higher CATE email
def combined_policy(row):
    if row['pt_mens'] == 1 and row['pt_womens'] == 1:
        return 'Mens E-Mail' if row['cate_mens'] >= row['cate_womens'] else 'Womens E-Mail'
    elif row['pt_mens'] == 1:
        return 'Mens E-Mail'
    elif row['pt_womens'] == 1:
        return 'Womens E-Mail'
    else:
        return 'Do Not Email'

df['policy_recommendation'] = df.apply(combined_policy, axis=1)

rec_counts = df['policy_recommendation'].value_counts()
print("=== Combined Policy Tree Recommendations ===")
for rec, cnt in rec_counts.items():
    print(f"  {rec}: {cnt:,} ({cnt/len(df):.1%})")
print()

# Compare to Stage 4 EV-based recommendations
stage4 = pd.read_csv('outputs/send_list.csv')
print("=== Stage 4 EV-based Recommendations (for comparison) ===")
for rec, cnt in stage4['recommendation'].value_counts().items():
    print(f"  {rec}: {cnt:,} ({cnt/len(df):.1%})")
print()

# Agreement rate
agree = (df['policy_recommendation'] == stage4['recommendation']).mean()
print(f"  Agreement between policy tree and EV ranking: {agree:.1%}")
print()

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
df[['policy_recommendation', 'pt_any', 'pt_mens', 'pt_womens']].to_csv(
    'outputs/policy_tree_recommendations.csv', index=False)
print("  Saved: outputs/policy_tree_recommendations.csv")

for name, model in [('pt_any', pt_any), ('pt_mens', pt_mens), ('pt_womens', pt_womens)]:
    with open(f'models/07_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

metadata = {
    'stage': 7,
    'model': 'PolicyTree (econml)',
    'max_depth': 3,
    'min_samples_leaf': 500,
    'cate_input': 'Stage 3 CausalForestDML CATE estimates',
    'trees': ['any_email', 'mens_email', 'womens_email'],
    'policy_counts': {k: int(v) for k, v in rec_counts.items()},
    'ev_blanket_send': float(round(ev_blanket, 4)),
    'ev_policy_tree': float(round(ev_tree, 4)),
    'pct_sent_tree': float(round(pct_sent, 2)),
    'agreement_with_stage4_ev': float(round(agree, 4)),
    'notes': (
        'PolicyTree learns decision rules that maximise expected CATE across the '
        'feature space. min_samples_leaf=500 enforces actionable leaf sizes. '
        'Combined policy: if both Mens and Womens trees say send, '
        'pick the email type with higher CATE for that customer. '
        'Rules saved as plain text in outputs/07_policy_rules.txt — '
        'deployable without a scoring pipeline.'
    )
}
with open('models/07_policy_tree_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("  Saved: models/07_policy_tree_metadata.json")
print()
print("=== Stage 7 complete ===")
