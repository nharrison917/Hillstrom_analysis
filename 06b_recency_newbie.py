# -*- coding: utf-8 -*-
"""
Stage 6b - Recency and Newbie Interaction Analysis

WHAT WE'RE EXPLORING (POSSIBLE_DIRECTIONS.md item 6)
-----------------------------------------------------
Two features appear in CATE importances but have never been cross-tabbed
against actual email responsiveness:

  recency  -- months since last purchase (1-12)
  newbie   -- 1 if customer acquired in last 12 months

Intuitions to test:
  1. Very recent purchasers (recency 1-2 months) may be organic rebuyers --
     they don't NEED the email nudge because they were about to buy anyway.
     If true, emailing them wastes spend with no incremental lift.

  2. Very lapsed customers (recency 10-12 months) may be unresponsive --
     the email doesn't reach them at the right moment.

  3. Newbies may respond differently depending on their catalog type and
     purchase history. A newbie with $400 history is different from one
     with $29.99.

  4. The recency x newbie combination may reveal a high-value acquisition
     segment: recently acquired customers with strong history who respond
     to email at above-average rates.

METHOD
------
For each segment (recency bucket / newbie group / cross-tab):
  - Compute actual conversion rate per treatment group
  - Compute actual mean spend per treatment group
  - Derive lift = treated - control
  - Flag segments where lift is notably positive or negative

This is descriptive analysis, not causal modeling. The randomised
treatment assignment means the treatment/control comparison within
each segment IS a valid causal estimate (no confounding within the RCT).

Run from project root with .venv active:
  source .venv/Scripts/activate
  python 06b_recency_newbie.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"


# ---------------------------------------------------------------------------
# 1. Load and prepare
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['log_history'] = np.log1p(df['history'])

# Email type label for readability
df['email_type'] = df['segment'].map({
    'Mens E-Mail': 'Mens Email',
    'Womens E-Mail': 'Womens Email',
    'No E-Mail': 'No Email (control)',
})

# Recency buckets -- chosen to reflect natural lifecycle stages:
#   1-2 months: recently active (potential organic rebuyers)
#   3-6 months: mid-lapse (prime email target window)
#   7-9 months: lapsing
#   10-12 months: heavily lapsed
recency_bins = [0, 2, 6, 9, 12]
recency_labels = ['1-2 mo (recent)', '3-6 mo (mid)', '7-9 mo (lapsing)', '10-12 mo (lapsed)']
df['recency_bucket'] = pd.cut(df['recency'], bins=recency_bins, labels=recency_labels)

print(f"  {len(df):,} customers loaded.")
print()
print("Recency distribution:")
print(df['recency_bucket'].value_counts().sort_index().to_string())
print()
print("Newbie distribution:")
print(df['newbie'].value_counts().sort_index().to_string())
print()


# ---------------------------------------------------------------------------
# 2. Lift calculation helper
# ---------------------------------------------------------------------------
def compute_lift(group_df):
    """
    Given a slice of the dataframe, compute lift metrics vs. control.
    Returns a dict with conversion rate, mean spend, and lift for each
    email type.
    """
    ctrl = group_df[group_df['segment'] == 'No E-Mail']
    mens = group_df[group_df['segment'] == 'Mens E-Mail']
    womens = group_df[group_df['segment'] == 'Womens E-Mail']

    def safe_mean(s, col):
        return s[col].mean() if len(s) > 0 else np.nan

    ctrl_conv = safe_mean(ctrl, 'conversion')
    ctrl_spend = safe_mean(ctrl, 'spend')

    return {
        'n_total': len(group_df),
        'n_ctrl': len(ctrl),
        'n_mens': len(mens),
        'n_womens': len(womens),
        'ctrl_conv_rate': ctrl_conv,
        'ctrl_mean_spend': ctrl_spend,
        'mens_conv_rate': safe_mean(mens, 'conversion'),
        'womens_conv_rate': safe_mean(womens, 'conversion'),
        'mens_conv_lift_pp': safe_mean(mens, 'conversion') - ctrl_conv,
        'womens_conv_lift_pp': safe_mean(womens, 'conversion') - ctrl_conv,
        'mens_spend_lift': safe_mean(mens, 'spend') - ctrl_spend,
        'womens_spend_lift': safe_mean(womens, 'spend') - ctrl_spend,
    }


# ---------------------------------------------------------------------------
# 3. Recency bucket analysis
# ---------------------------------------------------------------------------
print("=== RECENCY BUCKET ANALYSIS ===")
recency_results = []
for bucket in recency_labels:
    sub = df[df['recency_bucket'] == bucket]
    row = {'recency_bucket': bucket}
    row.update(compute_lift(sub))
    recency_results.append(row)

rdf = pd.DataFrame(recency_results)
print(rdf[['recency_bucket', 'n_total', 'ctrl_conv_rate',
           'mens_conv_lift_pp', 'womens_conv_lift_pp',
           'mens_spend_lift', 'womens_spend_lift']].to_string(index=False))
print()

# Flag notable findings
print("Key recency observations:")
for _, row in rdf.iterrows():
    bucket = row['recency_bucket']
    m_lift = row['mens_spend_lift']
    w_lift = row['womens_spend_lift']
    if pd.notna(m_lift) and m_lift < 0:
        print(f"  [!] {bucket}: Mens email NEGATIVE spend lift (${m_lift:.2f})")
    if pd.notna(w_lift) and w_lift < 0:
        print(f"  [!] {bucket}: Womens email NEGATIVE spend lift (${w_lift:.2f})")
    if pd.notna(m_lift) and m_lift > 1.5:
        print(f"  [+] {bucket}: Mens email strong lift (${m_lift:.2f})")
    if pd.notna(w_lift) and w_lift > 1.0:
        print(f"  [+] {bucket}: Womens email strong lift (${w_lift:.2f})")
print()


# ---------------------------------------------------------------------------
# 4. Newbie analysis
# ---------------------------------------------------------------------------
print("=== NEWBIE ANALYSIS ===")
newbie_results = []
for nv, label in [(0, 'Established (newbie=0)'), (1, 'New customer (newbie=1)')]:
    sub = df[df['newbie'] == nv]
    row = {'newbie': label}
    row.update(compute_lift(sub))
    newbie_results.append(row)

ndf = pd.DataFrame(newbie_results)
print(ndf[['newbie', 'n_total', 'ctrl_conv_rate',
           'mens_conv_lift_pp', 'womens_conv_lift_pp',
           'mens_spend_lift', 'womens_spend_lift']].to_string(index=False))
print()


# ---------------------------------------------------------------------------
# 5. Recency x Newbie cross-tab
# ---------------------------------------------------------------------------
print("=== RECENCY x NEWBIE CROSS-TAB (spend lift) ===")
cross_results = []
for bucket in recency_labels:
    for nv, nlabel in [(0, 'Established'), (1, 'Newbie')]:
        sub = df[(df['recency_bucket'] == bucket) & (df['newbie'] == nv)]
        row = {'recency_bucket': bucket, 'newbie': nlabel}
        row.update(compute_lift(sub))
        cross_results.append(row)

cdf = pd.DataFrame(cross_results)
print(cdf[['recency_bucket', 'newbie', 'n_total',
           'ctrl_conv_rate',
           'mens_spend_lift', 'womens_spend_lift']].to_string(index=False))
print()

# Flag strongest and weakest segments
cdf_valid = cdf.dropna(subset=['mens_spend_lift', 'womens_spend_lift'])
cdf_valid = cdf_valid[cdf_valid['n_total'] >= 100]  # ignore tiny cells

best_mens = cdf_valid.loc[cdf_valid['mens_spend_lift'].idxmax()]
worst_mens = cdf_valid.loc[cdf_valid['mens_spend_lift'].idxmin()]
best_womens = cdf_valid.loc[cdf_valid['womens_spend_lift'].idxmax()]
worst_womens = cdf_valid.loc[cdf_valid['womens_spend_lift'].idxmin()]

print("Strongest and weakest segments (min n=100):")
print(f"  Best Mens lift:    {best_mens['recency_bucket']} x {best_mens['newbie']}"
      f"  -> ${best_mens['mens_spend_lift']:.2f}  (n={int(best_mens['n_total']):,})")
print(f"  Worst Mens lift:   {worst_mens['recency_bucket']} x {worst_mens['newbie']}"
      f"  -> ${worst_mens['mens_spend_lift']:.2f}  (n={int(worst_mens['n_total']):,})")
print(f"  Best Womens lift:  {best_womens['recency_bucket']} x {best_womens['newbie']}"
      f"  -> ${best_womens['womens_spend_lift']:.2f}  (n={int(best_womens['n_total']):,})")
print(f"  Worst Womens lift: {worst_womens['recency_bucket']} x {worst_womens['newbie']}"
      f"  -> ${worst_womens['womens_spend_lift']:.2f}  (n={int(worst_womens['n_total']):,})")
print()


# ---------------------------------------------------------------------------
# 6. High-value micro-segment: Newbie x both_catalogs x Multichannel
# ---------------------------------------------------------------------------
# POSSIBLE_DIRECTIONS.md item 6 hypothesised this combination.
print("=== HIGH-VALUE MICRO-SEGMENT: Newbie + Both Catalogs ===")
df['both_catalogs'] = ((df['mens'] == 1) & (df['womens'] == 1)).astype(int)
micro = df[(df['newbie'] == 1) & (df['both_catalogs'] == 1)]
micro_mc = micro[micro['channel'] == 'Multichannel']

for label, sub in [('Newbie + both_catalogs', micro),
                   ('Newbie + both_catalogs + Multichannel', micro_mc)]:
    if len(sub) < 10:
        print(f"  {label}: too few customers (n={len(sub)})")
        continue
    row = compute_lift(sub)
    print(f"  {label} (n={row['n_total']:,}):")
    print(f"    Control conversion:  {row['ctrl_conv_rate']:.2%}")
    print(f"    Mens spend lift:     ${row['mens_spend_lift']:.2f}")
    print(f"    Womens spend lift:   ${row['womens_spend_lift']:.2f}")
print()


# ---------------------------------------------------------------------------
# 7. Visualisations
# ---------------------------------------------------------------------------
print("Building visualisations...")

# --- Chart 1: Recency bucket lift bars ---
fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Mens Email Spend Lift by Recency',
                    'Womens Email Spend Lift by Recency'],
    shared_yaxes=False,
)

colors_pos = '#2ca02c'
colors_neg = '#d62728'

for col_idx, (lift_col, title) in enumerate(
        [('mens_spend_lift', 'Mens'), ('womens_spend_lift', 'Womens')], 1):
    for _, row in rdf.iterrows():
        val = row[lift_col]
        if pd.isna(val):
            continue
        fig1.add_trace(go.Bar(
            x=[row['recency_bucket']],
            y=[val],
            marker_color=colors_pos if val >= 0 else colors_neg,
            showlegend=False,
            hovertemplate=(
                f"{row['recency_bucket']}<br>"
                f"n={int(row['n_total']):,}<br>"
                f"{title} spend lift: ${val:.2f}<extra></extra>"
            ),
        ), row=1, col=col_idx)

fig1.add_hline(y=0, line_dash='dash', line_color='gray')
fig1.update_layout(
    title=(
        'Email Spend Lift by Recency Bucket<br>'
        '<sup>Lift = mean spend (treated) - mean spend (control). '
        'Green = positive incremental revenue, red = negative.</sup>'
    ),
    height=420,
    template='plotly_white',
)
fig1.write_html('outputs/06b_recency_lift.html')
print("  Saved: outputs/06b_recency_lift.html")


# --- Chart 2: Recency x Newbie heatmap (Mens and Womens side by side) ---
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Mens Email Spend Lift', 'Womens Email Spend Lift'],
    shared_yaxes=True,
    horizontal_spacing=0.1,
)

for col_idx, lift_col in enumerate(['mens_spend_lift', 'womens_spend_lift'], 1):
    pivot = cdf.pivot(index='recency_bucket', columns='newbie',
                      values=lift_col)
    # Use consistent row order
    pivot = pivot.reindex(recency_labels)

    # Text annotations
    text_arr = []
    for rb in pivot.index:
        row_text = []
        for nb in pivot.columns:
            val = pivot.loc[rb, nb]
            n = cdf.loc[(cdf['recency_bucket'] == rb) &
                        (cdf['newbie'] == nb), 'n_total'].values
            n_str = f" (n={int(n[0]):,})" if len(n) > 0 else ""
            row_text.append(f'${val:.2f}{n_str}' if pd.notna(val) else 'n/a')
        text_arr.append(row_text)

    all_vals = [v for v in cdf[lift_col] if pd.notna(v)]
    zmin = min(all_vals) if all_vals else -1
    zmax = max(all_vals) if all_vals else 1

    hm = go.Heatmap(
        z=pivot.values.tolist(),
        x=list(pivot.columns),
        y=list(pivot.index),
        zmin=zmin,
        zmax=zmax,
        colorscale='RdYlGn',
        showscale=(col_idx == 2),
        colorbar=dict(title='Spend<br>Lift ($)'),
        text=text_arr,
        texttemplate='%{text}',
        hovertemplate='%{y} x %{x}<br>Spend lift: %{text}<extra></extra>',
    )
    fig2.add_trace(hm, row=1, col=col_idx)

fig2.update_xaxes(title_text='Customer type')
fig2.update_yaxes(title_text='Recency bucket', col=1)
fig2.update_layout(
    title=(
        'Email Spend Lift: Recency x Newbie Status<br>'
        '<sup>Red = negative lift (suppress), green = positive (target). '
        'Each cell: spend lift and customer count.</sup>'
    ),
    height=420,
    template='plotly_white',
)
fig2.write_html('outputs/06b_recency_newbie_heatmap.html')
print("  Saved: outputs/06b_recency_newbie_heatmap.html")


# --- Chart 3: Newbie comparison bar ---
fig3 = go.Figure()
for i, row in ndf.iterrows():
    for lift_col, label, color in [
            ('mens_spend_lift', 'Mens lift', '#2ca02c'),
            ('womens_spend_lift', 'Womens lift', '#9467bd')]:
        fig3.add_trace(go.Bar(
            name=f"{row['newbie']} - {label}",
            x=[row['newbie']],
            y=[row[lift_col]],
            marker_color=color,
            opacity=0.85 if 'Mens' in label else 0.6,
            hovertemplate=(
                f"{row['newbie']}<br>{label}: ${row[lift_col]:.2f}"
                f"<br>n={int(row['n_total']):,}<extra></extra>"
            ),
        ))

fig3.add_hline(y=0, line_dash='dash', line_color='gray')
fig3.update_layout(
    title='Email Spend Lift: Established vs New Customers',
    barmode='group',
    height=380,
    template='plotly_white',
    xaxis_title='Customer type',
    yaxis_title='Mean spend lift ($)',
)
fig3.write_html('outputs/06b_newbie_lift.html')
print("  Saved: outputs/06b_newbie_lift.html")
print()


# ---------------------------------------------------------------------------
# 8. Summary findings for NOTES.md / report
# ---------------------------------------------------------------------------
print("=== SUMMARY FINDINGS ===")
print()
print("Recency:")
for _, row in rdf.iterrows():
    print(f"  {row['recency_bucket']:25s}  "
          f"Mens lift=${row['mens_spend_lift']:+.2f}  "
          f"Womens lift=${row['womens_spend_lift']:+.2f}  "
          f"(ctrl conv={row['ctrl_conv_rate']:.2%})")
print()
print("Newbie:")
for _, row in ndf.iterrows():
    print(f"  {row['newbie']:30s}  "
          f"Mens lift=${row['mens_spend_lift']:+.2f}  "
          f"Womens lift=${row['womens_spend_lift']:+.2f}")
print()
print("Recency x Newbie (spend lift, Mens / Womens):")
for bucket in recency_labels:
    sub = cdf[cdf['recency_bucket'] == bucket]
    for _, row in sub.iterrows():
        print(f"  {bucket:25s} x {row['newbie']:12s}  "
              f"Mens=${row['mens_spend_lift']:+.2f}  "
              f"Womens=${row['womens_spend_lift']:+.2f}  "
              f"n={int(row['n_total']):,}")
    print()

print("=== Stage 6b complete ===")
