# -*- coding: utf-8 -*-
"""
Stage 7b - Policy Tree Hyperparameter Tuning

WHY THIS SCRIPT EXISTS
----------------------
Stage 7 fitted a single PolicyTree at depth=3, min_samples_leaf=500 by hand.
That's a reasonable starting point, but we don't know if it's the best setting
or whether it's overfitting to the training data. This script does it properly.

THREE IMPROVEMENTS OVER STAGE 7
---------------------------------

1. 5-fold cross-validation
   A single train/test split gives ONE noisy estimate of policy value.
   Flip the split and you get a different answer. k-fold averages across
   5 different splits, so the hyperparameter selection is stable.
   Stratified on treatment group so each fold has a proportional mix of
   Mens Email / Womens Email / No Email customers.

2. min_impurity_decrease sweep
   A node only splits if the improvement in policy value exceeds this
   threshold. This is pre-pruning: the tree won't chase tiny gains
   in the training data that likely won't generalise.
   We sweep [0.0, 0.01, 0.05].

3. Conservative training target (lower confidence bound) for Any Email tree
   The causal forest gives us a 90% confidence interval on CATE, not just
   a point estimate. We have cate_any_lb (5th percentile) from Stage 3.
   Training on the lower bound shifts the decision boundary: we only target
   customers where the CONSERVATIVE estimate of lift is positive.
   A customer whose mean CATE is +$0.20 but whose lower bound is -$0.40
   won't be targeted -- that's noise, not signal.
   Note: CI bounds were only computed for the "any email" comparison in
   Stage 3, so this is only available for the any-email tree.

SWEEP GRID
----------
depth in [2, 3, 4] x min_samples_leaf in [200, 500, 1000]
x min_impurity_decrease in [0.0, 0.01, 0.05]
= 27 combinations x 5 folds = 135 fits per tree type.
Runtime is fast -- PolicyTree is a shallow decision tree.

KEY METRIC: out-of-sample policy value
mean CATE of customers the tree recommends sending to, on held-out test folds.
Higher = the tree is selecting customers with more incremental revenue.
A plateau at higher depth signals we've hit the right complexity.

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 07b_policy_tree_tuning.py
"""

import json
import itertools
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from econml.policy import PolicyTree
import pickle

warnings.filterwarnings('ignore')

DATA_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
RANDOM_STATE = 42
N_FOLDS = 5

FEATURES = [
    'recency', 'log_history', 'mens', 'womens', 'both_catalogs', 'newbie',
    'history_segment_enc', 'zip_code_enc', 'channel_enc',
]

FEATURE_NAMES = [
    'recency', 'log_history', 'mens', 'womens', 'both_catalogs', 'newbie',
    'history_segment', 'zip_code', 'channel',
]

# Sweep grid
DEPTHS = [2, 3, 4]
MIN_LEAVES = [200, 500, 1000]
MIN_IMP_DECREASE = [0.0, 0.01, 0.05]


# ---------------------------------------------------------------------------
# 1. Load data and CATE scores
# ---------------------------------------------------------------------------
print("Loading data and Stage 3 CATE scores...")
df = pd.read_csv(DATA_FILE)
df_rev = pd.read_csv('outputs/scores_revenue.csv')
clusters = pd.read_csv('outputs/cluster_assignments_cate.csv')

df['log_history'] = np.log1p(df['history'])
df['both_catalogs'] = ((df['mens'] == 1) & (df['womens'] == 1)).astype(int)

le_hist = LabelEncoder()
le_zip = LabelEncoder()
le_chan = LabelEncoder()
df['history_segment_enc'] = le_hist.fit_transform(df['history_segment'].astype(str))
df['zip_code_enc'] = le_zip.fit_transform(df['zip_code'].astype(str))
df['channel_enc'] = le_chan.fit_transform(df['channel'].astype(str))

# Reverse-lookup maps for readable rule text
rev_hist = {int(v): k for k, v in
            zip(le_hist.classes_, le_hist.transform(le_hist.classes_))}
rev_zip = {int(v): k for k, v in
           zip(le_zip.classes_, le_zip.transform(le_zip.classes_))}
rev_chan = {int(v): k for k, v in
            zip(le_chan.classes_, le_chan.transform(le_chan.classes_))}

# Attach CATE scores
df['cate_any'] = df_rev['cate_any'].values
df['cate_any_lb'] = df_rev['cate_any_lb'].values   # 5th-percentile lower bound
df['cate_mens'] = (df_rev['cate_mens'].fillna(df_rev['cate_any']).values
                   if 'cate_mens' in df_rev.columns else df_rev['cate_any'].values)
df['cate_womens'] = (df_rev['cate_womens'].fillna(df_rev['cate_any']).values
                     if 'cate_womens' in df_rev.columns else df_rev['cate_any'].values)
df['cate_cluster'] = clusters['cate_cluster'].values

X = df[FEATURES].values
n_suppress_cluster = (df['cate_cluster'] == 0).sum()

print(f"  {len(df):,} customers | {X.shape[1]} features")
print(f"  CATE cluster C0 (negative uplift): {n_suppress_cluster:,} customers")
print(f"  cate_any mean: ${df['cate_any'].mean():.4f}  "
      f"(negative: {(df['cate_any'] < 0).sum():,})")
print(f"  cate_any_lb mean: ${df['cate_any_lb'].mean():.4f}  "
      f"(negative: {(df['cate_any_lb'] < 0).sum():,})")
print()


# ---------------------------------------------------------------------------
# 2. Cross-validation helper
# ---------------------------------------------------------------------------
def cv_policy_value(X, cate_train, cate_eval, strat_col,
                    max_depth, min_samples_leaf, min_impurity_decrease,
                    n_folds=N_FOLDS):
    """
    Run k-fold CV for a single hyperparameter combination.

    Parameters
    ----------
    X : array, shape (n, features)
        Customer features.
    cate_train : array, shape (n,)
        CATE values used as the training reward signal.
        For the conservative run this is cate_any_lb (lower bound).
        For standard runs this is the point estimate.
    cate_eval : array, shape (n,)
        CATE values used to MEASURE policy value on the test fold.
        Always the point estimate -- that's our best guess of true lift.
        Separating train vs eval targets lets us ask: "does training
        conservatively still select high-value customers?"
    strat_col : array
        Column to stratify folds on (treatment group).
    max_depth, min_samples_leaf, min_impurity_decrease :
        PolicyTree hyperparameters being swept.

    Returns
    -------
    (mean_policy_value, std_policy_value, mean_pct_sent)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_STATE)
    fold_values = []
    fold_pct_sent = []

    for train_idx, test_idx in skf.split(X, strat_col):
        X_train, X_test = X[train_idx], X[test_idx]
        cate_tr = cate_train[train_idx]
        cate_ev = cate_eval[test_idx]

        # PolicyTree expects a (n, n_arms) reward matrix.
        # Column 0 = control (no email) = 0 baseline.
        # Column 1 = treatment reward = CATE estimate.
        # The tree learns which leaf nodes have positive mean CATE (send)
        # vs negative (suppress).
        rewards_train = np.column_stack([np.zeros(len(cate_tr)), cate_tr])

        pt = PolicyTree(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=RANDOM_STATE,
        )
        pt.fit(X_train, rewards_train)

        # predict_value returns per-leaf mean reward for each arm.
        # Column 1 = treatment arm. Send if > 0 (tree prefers email over no-email).
        leaf_vals = pt.predict_value(X_test)[:, 1]
        send_mask = leaf_vals > 0

        if send_mask.sum() == 0:
            # Over-regularized: tree suppresses everyone.
            # Policy value is 0 (same as blanket no-send).
            fold_values.append(0.0)
            fold_pct_sent.append(0.0)
        elif (~send_mask).sum() == 0:
            # Under-regularized: tree sends to everyone.
            # Policy value = mean CATE of all test customers (same as blanket send).
            fold_values.append(float(cate_ev.mean()))
            fold_pct_sent.append(1.0)
        else:
            fold_values.append(float(cate_ev[send_mask].mean()))
            fold_pct_sent.append(float(send_mask.mean()))

    return (
        float(np.mean(fold_values)),
        float(np.std(fold_values)),
        float(np.mean(fold_pct_sent)),
    )


# ---------------------------------------------------------------------------
# 3. Full grid sweep
# ---------------------------------------------------------------------------
def run_sweep(cate_train_col, cate_eval_col, label):
    """
    Run all 27 hyperparameter combinations with 5-fold CV.
    Returns a DataFrame of results sorted by mean policy value (descending).
    """
    strat_col = df['segment'].values
    results = []
    total = len(DEPTHS) * len(MIN_LEAVES) * len(MIN_IMP_DECREASE)
    done = 0

    print(f"--- {label} ({total} combos x {N_FOLDS} folds) ---")

    for depth, min_leaf, min_imp in itertools.product(
            DEPTHS, MIN_LEAVES, MIN_IMP_DECREASE):
        mean_pv, std_pv, mean_pct = cv_policy_value(
            X, cate_train_col, cate_eval_col, strat_col,
            max_depth=depth,
            min_samples_leaf=min_leaf,
            min_impurity_decrease=min_imp,
        )
        results.append({
            'depth': depth,
            'min_samples_leaf': min_leaf,
            'min_impurity_decrease': min_imp,
            'mean_policy_value': mean_pv,
            'std_policy_value': std_pv,
            'mean_pct_sent': mean_pct,
        })
        done += 1
        print(f"  [{done:2d}/{total}] depth={depth}  min_leaf={min_leaf:4d}  "
              f"min_imp={min_imp:.2f}  ->  "
              f"policy_value=${mean_pv:.4f} +/- ${std_pv:.4f}  "
              f"({mean_pct:.1%} sent)")

    rdf = (pd.DataFrame(results)
           .sort_values('mean_policy_value', ascending=False)
           .reset_index(drop=True))

    best = rdf.iloc[0]
    print(f"\n  BEST: depth={int(best['depth'])}  "
          f"min_samples_leaf={int(best['min_samples_leaf'])}  "
          f"min_impurity_decrease={best['min_impurity_decrease']:.2f}  "
          f"-> policy_value=${best['mean_policy_value']:.4f} "
          f"(sent {best['mean_pct_sent']:.1%})")
    print()
    return rdf


print("=== HYPERPARAMETER SWEEP ===")
print()

sweep_any = run_sweep(
    df['cate_any'].values, df['cate_any'].values,
    'Any Email -- point estimate target')

sweep_any_lb = run_sweep(
    df['cate_any_lb'].values, df['cate_any'].values,
    'Any Email -- conservative lower-bound target')

sweep_mens = run_sweep(
    df['cate_mens'].values, df['cate_mens'].values,
    'Mens Email -- point estimate target')

sweep_womens = run_sweep(
    df['cate_womens'].values, df['cate_womens'].values,
    'Womens Email -- point estimate target')


# ---------------------------------------------------------------------------
# 4. Print comparison: point estimate vs conservative for Any Email
# ---------------------------------------------------------------------------
print("=== Point Estimate vs Conservative Target (Any Email) ===")
best_any = sweep_any.iloc[0]
best_any_lb = sweep_any_lb.iloc[0]
print(f"  Point estimate best:   depth={int(best_any['depth'])}, "
      f"min_leaf={int(best_any['min_samples_leaf'])}, "
      f"min_imp={best_any['min_impurity_decrease']:.2f} "
      f"-> ${best_any['mean_policy_value']:.4f} ({best_any['mean_pct_sent']:.1%} sent)")
print(f"  Conservative LB best:  depth={int(best_any_lb['depth'])}, "
      f"min_leaf={int(best_any_lb['min_samples_leaf'])}, "
      f"min_imp={best_any_lb['min_impurity_decrease']:.2f} "
      f"-> ${best_any_lb['mean_policy_value']:.4f} ({best_any_lb['mean_pct_sent']:.1%} sent)")
print()
# Interpretation: if conservative sends fewer customers but achieves higher
# policy value per sent customer, it's filtering noise. If policy value drops,
# we're cutting real signal too. Either result is informative.


# ---------------------------------------------------------------------------
# 5. Visualise: heatmaps (depth x min_samples_leaf, one panel per min_imp)
# ---------------------------------------------------------------------------
print("Building heatmap visualisations...")


def make_tuning_heatmap(sweep_df, label):
    """
    3-panel heatmap: one panel per min_impurity_decrease value.
    Axes: Y = max_depth, X = min_samples_leaf.
    Color = mean CV policy value ($).

    Reading this chart: find the cell with the darkest green. That combination
    produces the highest out-of-sample mean CATE among recommended-send customers.
    If values plateau across depths, shallower is safer (simpler rules, fewer
    false positives from overfitting to the training CATE surface).
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'min_impurity_decrease = {v}' for v in MIN_IMP_DECREASE],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    # Use a shared color scale so panels are comparable
    global_min = sweep_df['mean_policy_value'].min()
    global_max = sweep_df['mean_policy_value'].max()

    for col_idx, mid in enumerate(MIN_IMP_DECREASE):
        sub = sweep_df[sweep_df['min_impurity_decrease'] == mid].copy()
        pivot_val = sub.pivot(
            index='depth', columns='min_samples_leaf', values='mean_policy_value')
        pivot_pct = sub.pivot(
            index='depth', columns='min_samples_leaf', values='mean_pct_sent')

        # Cell text: policy value + % sent on second line
        text_arr = []
        for di in pivot_val.index:
            row_text = []
            for li in pivot_val.columns:
                pv = pivot_val.loc[di, li]
                ps = pivot_pct.loc[di, li]
                row_text.append(f'${pv:.3f}<br>{ps:.0%} sent')
            text_arr.append(row_text)

        hm = go.Heatmap(
            z=pivot_val.values.tolist(),
            x=[str(c) for c in pivot_val.columns],
            y=[str(r) for r in pivot_val.index],
            zmin=global_min,
            zmax=global_max,
            colorscale='RdYlGn',
            showscale=(col_idx == 2),
            colorbar=dict(title='Policy<br>Value ($)'),
            text=text_arr,
            texttemplate='%{text}',
            hovertemplate=(
                'depth=%{y}  min_leaf=%{x}<br>'
                'Policy value: %{text}<extra></extra>'
            ),
        )
        fig.add_trace(hm, row=1, col=col_idx + 1)

    fig.update_xaxes(title_text='min_samples_leaf', tickfont=dict(size=11))
    fig.update_yaxes(title_text='max_depth', col=1, tickfont=dict(size=11))

    fig.update_layout(
        title=(
            f'Policy Tree Tuning: {label}<br>'
            '<sup>Color = mean 5-fold CV policy value ($). '
            'Each cell shows value and % of customers sent. '
            'Higher value = better targeting of incremental revenue.</sup>'
        ),
        height=380,
        template='plotly_white',
        font=dict(size=12),
    )
    return fig


fig_any = make_tuning_heatmap(sweep_any, 'Any Email (point estimate target)')
fig_any.write_html('outputs/07b_tuning_heatmap_any.html')
print("  Saved: outputs/07b_tuning_heatmap_any.html")

fig_any_lb = make_tuning_heatmap(sweep_any_lb,
                                  'Any Email (conservative lower-bound target)')
fig_any_lb.write_html('outputs/07b_tuning_heatmap_any_lb.html')
print("  Saved: outputs/07b_tuning_heatmap_any_lb.html")

fig_mens = make_tuning_heatmap(sweep_mens, 'Mens Email')
fig_mens.write_html('outputs/07b_tuning_heatmap_mens.html')
print("  Saved: outputs/07b_tuning_heatmap_mens.html")

fig_womens = make_tuning_heatmap(sweep_womens, 'Womens Email')
fig_womens.write_html('outputs/07b_tuning_heatmap_womens.html')
print("  Saved: outputs/07b_tuning_heatmap_womens.html")
print()


# ---------------------------------------------------------------------------
# 6. Select best parameters and refit on the full dataset
# ---------------------------------------------------------------------------
# We pick the best from the standard point-estimate sweep for each tree type.
# The conservative (lower-bound) sweep is reported separately as a comparison.
#
# Tie-breaking rule: if two combos are within $0.001 of each other in policy
# value, prefer the one with lower depth (simpler, more explainable rules).

def select_best(sweep_df, tol=0.001):
    """
    Return best row, with tie-breaking in favour of shallower depth.
    """
    best_val = sweep_df['mean_policy_value'].iloc[0]
    candidates = sweep_df[
        sweep_df['mean_policy_value'] >= best_val - tol
    ].sort_values('depth')
    return candidates.iloc[0]


best_any_params = select_best(sweep_any)
best_mens_params = select_best(sweep_mens)
best_womens_params = select_best(sweep_womens)

print("=== SELECTED PARAMETERS (full-data refit) ===")
for label, row in [('Any', best_any_params),
                   ('Mens', best_mens_params),
                   ('Womens', best_womens_params)]:
    print(f"  {label}: depth={int(row['depth'])}  "
          f"min_leaf={int(row['min_samples_leaf'])}  "
          f"min_imp={row['min_impurity_decrease']:.2f}  "
          f"| policy_value=${row['mean_policy_value']:.4f}  "
          f"({row['mean_pct_sent']:.1%} sent)")
print()


def refit_full(cate_values, best_params, label):
    """Fit a PolicyTree on the full dataset with the best hyperparameters."""
    print(f"  Refitting {label} on full data "
          f"(depth={int(best_params['depth'])}, "
          f"min_leaf={int(best_params['min_samples_leaf'])}, "
          f"min_imp={best_params['min_impurity_decrease']:.2f})...")
    pt = PolicyTree(
        max_depth=int(best_params['depth']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        min_impurity_decrease=float(best_params['min_impurity_decrease']),
        random_state=RANDOM_STATE,
    )
    rewards = np.column_stack([np.zeros(len(cate_values)), cate_values])
    pt.fit(X, rewards)
    print(f"    Done. Actual depth: {pt.get_depth()} | Leaves: {pt.tree_.n_leaves}")
    return pt


print("Refitting best models on full dataset...")
pt_any = refit_full(df['cate_any'].values, best_any_params, 'Any Email')
pt_mens = refit_full(df['cate_mens'].values, best_mens_params, 'Mens Email')
pt_womens = refit_full(df['cate_womens'].values, best_womens_params, 'Womens Email')
print()


# ---------------------------------------------------------------------------
# 7. Extract human-readable rules
# ---------------------------------------------------------------------------
def tree_to_rules(pt, feature_names, rev_maps, label):
    """
    Walk the fitted policy tree and produce plain-English if/then rules.
    Each leaf shows the rule path, recommended action, and mean CATE.

    This is the output that can go directly into a business decision memo:
    "If history <= $30 AND recency > 6 months AND not a new customer, suppress."
    No scoring pipeline required.
    """
    tree = pt.tree_
    lines = [f"\n=== Policy Tree: {label} ==="]
    lines.append(f"  Depth: {pt.get_depth()} | Leaves: {tree.n_leaves}")
    lines.append("")

    rev_map_lookup = {
        6: rev_maps['history_segment'],
        7: rev_maps['zip_code'],
        8: rev_maps['channel'],
    }

    def fmt_threshold(feat_idx, threshold):
        if feat_idx in rev_map_lookup:
            below = [v for k, v in rev_map_lookup[feat_idx].items()
                     if k <= threshold]
            above = [v for k, v in rev_map_lookup[feat_idx].items()
                     if k > threshold]
            return str(below), str(above)
        elif feature_names[feat_idx] == 'log_history':
            val = np.expm1(threshold)
            return f"history <= ${val:.0f}", f"history > ${val:.0f}"
        else:
            return (f"{feature_names[feat_idx]} <= {threshold:.2f}",
                    f"{feature_names[feat_idx]} > {threshold:.2f}")

    def recurse(node_id, depth, path):
        indent = "  " * (depth + 1)
        feat = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        value = tree.value[node_id]

        if feat < 0:  # leaf node
            # value shape is (n_arms, 1); arm 1 = treatment
            mean_cate = (float(value[1][0]) if len(value) > 1
                         else float(value[0][0]))
            action = "SEND EMAIL" if mean_cate > 0 else "DO NOT EMAIL"
            rule = " AND ".join(path) if path else "(all customers)"
            lines.append(f"{indent}IF {rule}")
            lines.append(f"{indent}  -> {action}  (mean CATE: ${mean_cate:.4f})")
            lines.append("")
            return

        fname = feature_names[feat]
        left_desc, right_desc = fmt_threshold(feat, thresh)

        left_path = (path + [left_desc if isinstance(left_desc, str)
                              else f"{fname} in {left_desc}"])
        right_path = (path + [right_desc if isinstance(right_desc, str)
                               else f"{fname} in {right_desc}"])

        recurse(tree.children_left[node_id], depth + 1, left_path)
        recurse(tree.children_right[node_id], depth + 1, right_path)

    recurse(0, 0, [])
    return "\n".join(lines)


rev_maps = {
    'history_segment': rev_hist,
    'zip_code': rev_zip,
    'channel': rev_chan,
}

rules_any = tree_to_rules(pt_any, FEATURE_NAMES, rev_maps,
                           'Any Email vs No Email (tuned)')
rules_mens = tree_to_rules(pt_mens, FEATURE_NAMES, rev_maps,
                            'Mens Email vs No Email (tuned)')
rules_womens = tree_to_rules(pt_womens, FEATURE_NAMES, rev_maps,
                              'Womens Email vs No Email (tuned)')

print(rules_any)
print(rules_mens)
print(rules_womens)

with open('outputs/07_policy_rules.txt', 'w', encoding='utf-8') as f:
    header = (
        "POLICY TREE RULES - TUNED (Stage 7b)\n"
        f"Any Email: depth={int(best_any_params['depth'])}, "
        f"min_leaf={int(best_any_params['min_samples_leaf'])}, "
        f"min_imp={best_any_params['min_impurity_decrease']:.2f}\n"
        f"Mens Email: depth={int(best_mens_params['depth'])}, "
        f"min_leaf={int(best_mens_params['min_samples_leaf'])}, "
        f"min_imp={best_mens_params['min_impurity_decrease']:.2f}\n"
        f"Womens Email: depth={int(best_womens_params['depth'])}, "
        f"min_leaf={int(best_womens_params['min_samples_leaf'])}, "
        f"min_imp={best_womens_params['min_impurity_decrease']:.2f}\n"
        "=" * 60 + "\n"
    )
    f.write(header)
    f.write(rules_any + "\n")
    f.write(rules_mens + "\n")
    f.write(rules_womens + "\n")
print("Saved: outputs/07_policy_rules.txt")
print()


# ---------------------------------------------------------------------------
# 8. Apply trees, build combined policy, suppression gap analysis
# ---------------------------------------------------------------------------
def get_policy_actions(pt, X):
    """Return send/no-send array for each customer (1=send, 0=suppress)."""
    leaf_vals = pt.predict_value(X)[:, 1]
    return (leaf_vals > 0).astype(int)


policy_any = get_policy_actions(pt_any, X)
policy_mens = get_policy_actions(pt_mens, X)
policy_womens = get_policy_actions(pt_womens, X)

df['pt_any'] = policy_any
df['pt_mens'] = policy_mens
df['pt_womens'] = policy_womens


def combined_policy(row):
    """
    Combine Mens and Womens trees into a single recommendation.
    If both say send: pick the email type with the higher individual CATE.
    If one says send: send that type.
    If neither: Do Not Email.
    """
    if row['pt_mens'] == 1 and row['pt_womens'] == 1:
        return ('Mens E-Mail' if row['cate_mens'] >= row['cate_womens']
                else 'Womens E-Mail')
    elif row['pt_mens'] == 1:
        return 'Mens E-Mail'
    elif row['pt_womens'] == 1:
        return 'Womens E-Mail'
    else:
        return 'Do Not Email'


df['policy_recommendation'] = df.apply(combined_policy, axis=1)

rec_counts = df['policy_recommendation'].value_counts()
print("=== Combined Policy Recommendations (tuned trees) ===")
for rec, cnt in rec_counts.items():
    print(f"  {rec}: {cnt:,} ({cnt / len(df):.1%})")
print()


# --- Suppression gap analysis ---
# Key question from POSSIBLE_DIRECTIONS.md item 7:
# - 17,233 customers in CATE cluster C0 have negative expected value from emailing.
# - The Stage 7 policy tree only cleanly suppressed 2,127.
# - Has tuning closed that gap?
#
# We also flag the "leakage" issue: customers suppressed for Mens email but
# still receiving Womens email, even though their Womens CATE is also negative.

n_dne = (df['policy_recommendation'] == 'Do Not Email').sum()
n_c0 = n_suppress_cluster  # 17,233 negative-CATE cluster

# Among C0 cluster customers: how many does the tuned tree suppress?
c0_mask = df['cate_cluster'] == 0
c0_suppressed = ((df.loc[c0_mask, 'policy_recommendation'] == 'Do Not Email')
                 .sum())
c0_still_sent = c0_mask.sum() - c0_suppressed

# Leakage: customers where pt_mens=0 (suppress) but pt_womens=1 (send)
# and their actual womens CATE is also negative
mens_suppress_mask = (df['pt_mens'] == 0) & (df['pt_womens'] == 1)
mens_suppress_neg_womens = (mens_suppress_mask &
                             (df['cate_womens'] < 0)).sum()

# Stage 7 baseline for comparison
stage7_dne = 2127

print("=== SUPPRESSION GAP ANALYSIS ===")
print(f"  CATE cluster C0 (negative uplift):     {n_c0:,} customers")
print(f"  Stage 7 (untuned) Do Not Email:        {stage7_dne:,}")
print(f"  Stage 7b (tuned) Do Not Email:         {n_dne:,}")
print()
print(f"  C0 customers suppressed by tuned tree: {c0_suppressed:,} "
      f"({c0_suppressed / n_c0:.1%} of C0)")
print(f"  C0 customers still being sent email:   {c0_still_sent:,} "
      f"({c0_still_sent / n_c0:.1%} of C0)")
print()
print(f"  Remaining gap (C0 - tuned DNE):        "
      f"{n_c0 - n_dne:,} customers")
print()
print("  Leakage check (Blind Spot #7 from POSSIBLE_DIRECTIONS.md):")
print(f"  Customers in Mens-suppress but sent Womens: {mens_suppress_mask.sum():,}")
print(f"  Of those, with negative Womens CATE too:    {mens_suppress_neg_womens:,}")
if mens_suppress_neg_womens > 0:
    print(f"  These {mens_suppress_neg_womens:,} customers should ideally be caught by "
          f"a CATE C0 upstream filter (see POSSIBLE_DIRECTIONS.md item 7).")
print()

# --- Policy value comparison ---
cate = df['cate_any'].values
send_mask = policy_any == 1
ev_blanket = cate.mean()
ev_tree = cate[send_mask].mean() if send_mask.sum() > 0 else 0.0
pct_sent = send_mask.mean()
stage7_ev = None  # compare to metadata if available

print("=== POLICY VALUE COMPARISON (Any Email) ===")
print(f"  Blanket send (everyone):  mean CATE = ${ev_blanket:.4f}")
print(f"  No send:                  mean CATE = $0.0000")
print(f"  Stage 7b policy tree:     mean CATE = ${ev_tree:.4f}  "
      f"(sends {send_mask.sum():,} = {pct_sent:.1%})")
try:
    with open('models/07_policy_tree_metadata.json') as f:
        meta7 = json.load(f)
    print(f"  Stage 7 (untuned) tree:   mean CATE = "
          f"${meta7['ev_policy_tree']:.4f}  "
          f"({meta7['pct_sent_tree']:.1f}% sent)")
except Exception:
    pass
print()

# --- Compare to Stage 4 EV recommendations ---
stage4 = pd.read_csv('outputs/send_list.csv')
agree = (df['policy_recommendation'] == stage4['recommendation']).mean()
print(f"  Agreement with Stage 4 EV-based ranking: {agree:.1%}")
print()


# ---------------------------------------------------------------------------
# 8b. C0 upstream suppression filter (POSSIBLE_DIRECTIONS.md item 7 fix)
# ---------------------------------------------------------------------------
# The policy tree has a leakage problem: 292 customers are suppressed by the
# Mens tree but still routed to Womens email, even though their Womens CATE
# is also negative. The trees are fitted independently so they don't know about
# each other's suppression decisions.
#
# Fix: apply the CATE cluster C0 assignment (Stage 6) as an upstream gate.
# If a customer is in C0 (negative mean CATE), suppress regardless of what
# either tree recommends.
#
# This is a design choice, not a modeling step. We're saying:
#   "Trust the causal forest's cluster assignment over the individual tree rules
#    for customers where the model already told us the email has negative expected value."
#
# We show both versions (rule-only and C0-filtered) so the trade-off is explicit.
# The rule-only version (2,127 DNE) is deployable without any model infrastructure.
# The C0-filtered version (17,233 DNE) requires the Stage 3/6 scoring pipeline.

df['policy_c0_filtered'] = df['policy_recommendation'].copy()
df.loc[df['cate_cluster'] == 0, 'policy_c0_filtered'] = 'Do Not Email'

rec_c0 = df['policy_c0_filtered'].value_counts()
leakage_fixed = (
    (df['policy_recommendation'] != 'Do Not Email') &
    (df['policy_c0_filtered'] == 'Do Not Email')
).sum()

# Policy value under C0 filter: mean CATE of customers still sent
sent_c0_mask = df['policy_c0_filtered'] != 'Do Not Email'
ev_c0_filtered = df.loc[sent_c0_mask, 'cate_any'].mean()

print("=== C0 UPSTREAM SUPPRESSION FILTER (POSSIBLE_DIRECTIONS.md item 7) ===")
print("  Rule-only policy (no C0 filter):")
for rec, cnt in rec_counts.items():
    print(f"    {rec}: {cnt:,} ({cnt / len(df):.1%})")
print()
print("  C0-filtered policy (C0 cluster -> Do Not Email):")
for rec, cnt in rec_c0.items():
    print(f"    {rec}: {cnt:,} ({cnt / len(df):.1%})")
print()
print(f"  Customers moved to DNE by C0 filter: {leakage_fixed:,}")
print(f"  Of those, leakage customers fixed:   {mens_suppress_neg_womens:,}")
print()
print(f"  Policy value (mean CATE of sent customers):")
print(f"    Rule-only:    ${df.loc[df['policy_recommendation'] != 'Do Not Email', 'cate_any'].mean():.4f}")
print(f"    C0-filtered:  ${ev_c0_filtered:.4f}")
print()
print("  Note: C0-filtered requires the Stage 3/6 scoring pipeline to deploy.")
print("  Rule-only requires no model infrastructure -- it is a list of if/then conditions.")
print("  Both versions are saved to outputs/.")
print()


# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------
print("Saving outputs...")

# Recommendations CSV — both rule-only and C0-filtered versions
df[['policy_recommendation', 'policy_c0_filtered',
    'pt_any', 'pt_mens', 'pt_womens',
    'cate_any', 'cate_mens', 'cate_womens',
    'cate_cluster']].to_csv(
    'outputs/policy_tree_recommendations.csv', index=False)
print("  Saved: outputs/policy_tree_recommendations.csv")

# Sweep results (useful for the report / portfolio documentation)
sweep_any.to_csv('outputs/07b_sweep_any.csv', index=False)
sweep_any_lb.to_csv('outputs/07b_sweep_any_lb.csv', index=False)
sweep_mens.to_csv('outputs/07b_sweep_mens.csv', index=False)
sweep_womens.to_csv('outputs/07b_sweep_womens.csv', index=False)
print("  Saved: outputs/07b_sweep_*.csv")

# Models
for name, model in [('07b_pt_any', pt_any),
                    ('07b_pt_mens', pt_mens),
                    ('07b_pt_womens', pt_womens)]:
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
print("  Saved: models/07b_pt_*.pkl")

# Metadata
metadata = {
    'stage': '7b',
    'model': 'PolicyTree (econml) - tuned via 5-fold CV',
    'sweep': {
        'depths': DEPTHS,
        'min_samples_leaf': MIN_LEAVES,
        'min_impurity_decrease': MIN_IMP_DECREASE,
        'n_folds': N_FOLDS,
        'n_combinations': len(DEPTHS) * len(MIN_LEAVES) * len(MIN_IMP_DECREASE),
    },
    'best_params': {
        'any_email': {
            'depth': int(best_any_params['depth']),
            'min_samples_leaf': int(best_any_params['min_samples_leaf']),
            'min_impurity_decrease': float(best_any_params['min_impurity_decrease']),
            'cv_policy_value': float(round(best_any_params['mean_policy_value'], 4)),
            'cv_pct_sent': float(round(best_any_params['mean_pct_sent'], 4)),
        },
        'mens_email': {
            'depth': int(best_mens_params['depth']),
            'min_samples_leaf': int(best_mens_params['min_samples_leaf']),
            'min_impurity_decrease': float(best_mens_params['min_impurity_decrease']),
            'cv_policy_value': float(round(best_mens_params['mean_policy_value'], 4)),
            'cv_pct_sent': float(round(best_mens_params['mean_pct_sent'], 4)),
        },
        'womens_email': {
            'depth': int(best_womens_params['depth']),
            'min_samples_leaf': int(best_womens_params['min_samples_leaf']),
            'min_impurity_decrease': float(best_womens_params['min_impurity_decrease']),
            'cv_policy_value': float(round(best_womens_params['mean_policy_value'], 4)),
            'cv_pct_sent': float(round(best_womens_params['mean_pct_sent'], 4)),
        },
    },
    'suppression_gap': {
        'cate_c0_cluster_size': int(n_c0),
        'stage7_dne': stage7_dne,
        'stage7b_dne': int(n_dne),
        'c0_suppressed_by_tree': int(c0_suppressed),
        'c0_still_sent': int(c0_still_sent),
        'leakage_mens_suppress_sent_womens': int(mens_suppress_mask.sum()),
        'leakage_both_negative_cate': int(mens_suppress_neg_womens),
    },
    'policy_value': {
        'blanket_send': float(round(ev_blanket, 4)),
        'policy_tree': float(round(ev_tree, 4)),
        'pct_sent': float(round(pct_sent * 100, 2)),
        'agreement_with_stage4_ev': float(round(agree, 4)),
    },
    'recommendations_rule_only': {k: int(v) for k, v in rec_counts.items()},
    'recommendations_c0_filtered': {k: int(v) for k, v in rec_c0.items()},
    'notes': (
        'Tuned via 5-fold stratified CV (stratified on treatment group). '
        'Best params selected by mean CV policy value with tie-breaking '
        'in favour of shallower depth (simpler, more explainable rules). '
        'Also ran conservative lower-bound sweep for Any Email tree '
        '(trains on 5th-percentile CATE from Stage 3 causal forest, '
        'evaluates on point estimate). '
        'Rules saved in outputs/07_policy_rules.txt. '
        'Combined policy: if both Mens and Womens trees say send, '
        'pick the type with higher individual CATE.'
    ),
}

with open('models/07b_policy_tree_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("  Saved: models/07b_policy_tree_metadata.json")

print()
print("=== Stage 7b complete ===")
