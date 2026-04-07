# -*- coding: utf-8 -*-
"""
08_app.py -- Hillstrom Uplift Modeling: Interactive Streamlit Dashboard

Two audience modes (sidebar toggle):
  Stakeholder view: cluster personas, policy summary, plain-language findings
  Technical view:   CATE distributions, policy tree rules, feature importances, limitations

Run from project root with .venv active:
  streamlit run 08_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hillstrom Uplift Analysis",
    page_icon="assets/favicon.ico" if False else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = (
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_FILE)
    scores = pd.read_csv("outputs/scores_revenue.csv")
    policy = pd.read_csv("outputs/policy_tree_recommendations.csv")
    cluster_profiles = pd.read_csv("outputs/cate_cluster_profiles.csv")
    return raw, scores, policy, cluster_profiles


@st.cache_data
def load_policy_rules():
    with open("outputs/07_policy_rules.txt", "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_html(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data
def build_recency_newbie_data(raw):
    """Reproduce the recency x newbie cross-tab from Stage 6b."""
    df = raw.copy()
    recency_bins = [0, 2, 6, 9, 12]
    recency_labels = ["1-2 mo (recent)", "3-6 mo (mid)", "7-9 mo (lapsing)", "10-12 mo (lapsed)"]
    df["recency_bucket"] = pd.cut(df["recency"], bins=recency_bins, labels=recency_labels)

    rows = []
    for bucket in recency_labels:
        for nv, nlabel in [(0, "Established"), (1, "Newbie")]:
            sub = df[(df["recency_bucket"] == bucket) & (df["newbie"] == nv)]
            ctrl = sub[sub["segment"] == "No E-Mail"]
            mens = sub[sub["segment"] == "Mens E-Mail"]
            womens = sub[sub["segment"] == "Womens E-Mail"]
            ctrl_spend = ctrl["spend"].mean() if len(ctrl) > 0 else np.nan
            rows.append({
                "recency_bucket": bucket,
                "newbie": nlabel,
                "n_total": len(sub),
                "mens_spend_lift": mens["spend"].mean() - ctrl_spend if len(mens) > 0 else np.nan,
                "womens_spend_lift": womens["spend"].mean() - ctrl_spend if len(womens) > 0 else np.nan,
            })
    return pd.DataFrame(rows), recency_labels


raw, scores, policy, cluster_profiles = load_data()
rules_text = load_policy_rules()
cdf, recency_labels = build_recency_newbie_data(raw)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Hillstrom\nUplift Analysis")
    st.caption("Causal ML on 64,000 customers\nRandomized email marketing experiment")
    st.divider()
    view = st.radio(
        "Audience",
        ["Stakeholder View", "Technical View"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "**Dataset:** Hillstrom MineThatData (2008)\n\n"
        "**Pipeline:** T-Learner + Causal Forest + Policy Tree\n\n"
        "**Goal:** Maximize incremental revenue, not just conversion rate"
    )

# ---------------------------------------------------------------------------
# Shared chart helpers
# ---------------------------------------------------------------------------
CLUSTER_COLORS = {
    0: "#ef4444",   # red -- suppress
    1: "#22c55e",   # green -- high value
    2: "#3b82f6",   # blue -- moderate
    3: "#8b5cf6",   # purple -- moderate
}

CLUSTER_ACTIONS = {
    0: "Do Not Email",
    1: "Priority: Send Either",
    2: "Send Mens Preferred",
    3: "Send Either",
}


def make_recency_newbie_heatmap(cdf, recency_labels):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Mens Email Spend Lift", "Womens Email Spend Lift"],
        shared_yaxes=True,
        horizontal_spacing=0.12,
    )
    for col_idx, lift_col in enumerate(["mens_spend_lift", "womens_spend_lift"], 1):
        pivot = cdf.pivot(index="recency_bucket", columns="newbie", values=lift_col)
        pivot = pivot.reindex(recency_labels)

        text_arr = []
        for rb in pivot.index:
            row_text = []
            for nb in pivot.columns:
                val = pivot.loc[rb, nb]
                n = cdf.loc[(cdf["recency_bucket"] == rb) & (cdf["newbie"] == nb), "n_total"].values
                n_str = f"<br>n={int(n[0]):,}" if len(n) > 0 else ""
                row_text.append(f"${val:.2f}{n_str}" if pd.notna(val) else "n/a")
            text_arr.append(row_text)

        all_vals = cdf[lift_col].dropna().tolist()
        zmin, zmax = min(all_vals), max(all_vals)

        fig.add_trace(
            go.Heatmap(
                z=pivot.values.tolist(),
                x=list(pivot.columns),
                y=list(pivot.index),
                zmin=zmin, zmax=zmax,
                colorscale="RdYlGn",
                showscale=(col_idx == 2),
                colorbar=dict(title="Spend<br>Lift ($)"),
                text=text_arr,
                texttemplate="%{text}",
                hovertemplate="%{y} x %{x}<br>Spend lift: %{text}<extra></extra>",
            ),
            row=1, col=col_idx,
        )

    fig.update_xaxes(title_text="Customer type")
    fig.update_yaxes(title_text="Recency bucket", col=1)
    fig.update_layout(
        height=380,
        margin=dict(t=60, b=20),
        template="plotly_white",
    )
    return fig


def make_policy_comparison_chart(policy):
    """Bar chart: Blanket vs Rule-only vs C0-filtered policy outcomes."""
    labels = ["Blanket Send", "Rule-Only Policy", "C0-Filtered Policy"]
    sent_counts = [64000, 61873, 47111]
    sent_pct = [100.0, 96.7, 73.4]
    mean_cate = [0.63, 0.70, 1.09]
    total_rev = [n * pv for n, pv in zip(sent_counts, mean_cate)]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "<b>% Emailed</b>",
            "<b>Mean CATE / Customer</b>",
            "<b>Est. Total Revenue</b>",
        ],
        horizontal_spacing=0.08,
    )
    colors = ["#94a3b8", "#60a5fa", "#22c55e"]

    fig.add_trace(
        go.Bar(x=labels, y=sent_pct, marker_color=colors, showlegend=False,
               text=[f"{v:.1f}%" for v in sent_pct], textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=mean_cate, marker_color=colors, showlegend=False,
               text=[f"${v:.2f}" for v in mean_cate], textposition="outside"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=labels, y=total_rev, marker_color=colors, showlegend=False,
               text=[f"${v:,.0f}" for v in total_rev], textposition="outside"),
        row=1, col=3,
    )
    fig.update_layout(height=380, template="plotly_white", margin=dict(t=50, b=20))
    fig.update_yaxes(range=[0, 115], col=1)
    fig.update_yaxes(range=[0, 1.35], col=2)
    fig.update_yaxes(range=[0, 62000], col=3)
    # Reduce annotation font size so titles don't overlap at moderate window widths
    for annotation in fig.layout.annotations:
        annotation.font.size = 12
    return fig


def make_cate_density(scores):
    """Overlapping KDE density curves for each email type with mean lines."""
    from scipy.stats import gaussian_kde

    email_types = [
        ("cate_any",    "Any Email",    "#60a5fa", 0.15),
        ("cate_mens",   "Mens Email",   "#34d399", 0.15),
        ("cate_womens", "Womens Email", "#f97316", 0.15),
    ]

    # Clip to a readable range — extreme outliers compress the bulk of the distribution
    x_range = (-8, 12)
    x_vals = np.linspace(*x_range, 500)

    fig = go.Figure()

    for col, label, color, opacity in email_types:
        data = scores[col].dropna().clip(*x_range).values
        kde = gaussian_kde(data, bw_method=0.3)
        y_vals = kde(x_vals)
        mean_val = scores[col].mean()

        # Filled density curve
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            name=label,
            mode="lines",
            fill="tozeroy",
            fillcolor=color.replace(")", f", {opacity})").replace("rgb", "rgba")
                       if color.startswith("rgb") else
                       f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{opacity})",
            line=dict(color=color, width=2),
        ))

        # Vertical mean line
        mean_density = float(kde([mean_val])[0])
        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val],
            y=[0, mean_density],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            showlegend=False,
            hovertemplate=f"{label} mean: ${mean_val:.2f}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                  annotation_text="CATE = 0", annotation_position="top right",
                  annotation_font_size=10)

    fig.update_layout(
        title="CATE Score Distribution by Email Type",
        xaxis_title="Estimated Incremental Spend per Customer ($)",
        yaxis_title="Density",
        height=420,
        template="plotly_white",
        showlegend=True,
        xaxis=dict(range=x_range),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_cluster_cate_bar(cluster_profiles):
    """Bar chart of mean CATE by cluster."""
    fig = go.Figure()
    for _, row in cluster_profiles.iterrows():
        c = int(row["cate_cluster"])
        fig.add_trace(go.Bar(
            x=[f"C{c}: {row['persona']}"],
            y=[row["cate_any"]],
            marker_color=CLUSTER_COLORS[c],
            name=f"C{c}",
            showlegend=False,
            text=f"${row['cate_any']:.2f}",
            textposition="outside",
            hovertemplate=(
                f"Cluster C{c}<br>n={int(row['n']):,}<br>"
                f"Mean CATE: ${row['cate_any']:.2f}<br>"
                f"Persona: {row['persona']}<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Mean CATE by Customer Segment",
        yaxis_title="Mean Incremental Spend ($)",
        height=360,
        template="plotly_white",
    )
    return fig


def make_send_breakdown_chart(policy):
    """Donut chart of C0-filtered send recommendations."""
    c0_filtered = policy["policy_c0_filtered"].value_counts()
    labels = list(c0_filtered.index)
    values = list(c0_filtered.values)
    colors = []
    for lbl in labels:
        if "Do Not" in lbl or "DNE" in lbl or "No" in lbl:
            colors.append("#ef4444")
        elif "Mens" in lbl:
            colors.append("#3b82f6")
        else:
            colors.append("#8b5cf6")

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:,} customers<extra></extra>",
    ))
    fig.update_layout(
        title="C0-Filtered Policy: Email Type Breakdown",
        height=360,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# STAKEHOLDER VIEW
# ---------------------------------------------------------------------------
if view == "Stakeholder View":
    st.title("Email Marketing: Who Should We Send To?")
    st.caption(
        "Causal ML analysis of 64,000 customers from a randomized email marketing experiment. "
        "The goal is incremental revenue — not just conversion rate. Customers who would "
        "have bought without an email are excluded from the send list."
    )

    # --- Hero metrics ---
    c0_count = int(cluster_profiles.loc[cluster_profiles["cate_cluster"] == 0, "n"].values[0])
    to_email = 64000 - c0_count
    policy_uplift = round((1.09 - 0.63) / 0.63 * 100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers Analyzed", "64,000", help="Full Hillstrom dataset, 2008 RCT")
    col2.metric("Recommended to Email", f"{to_email:,}", help="After suppressing C0 cluster (C0-filtered policy)")
    col3.metric("Do Not Email", f"{c0_count:,}", help="Negative expected uplift — organic buyers")
    col4.metric(
        "Est. Revenue Gain vs Blanket Send",
        "~$11,100",
        delta=f"+{policy_uplift}% per customer",
        help="C0-filtered policy: ~$51,400 vs blanket send ~$40,300",
    )

    st.divider()

    # --- Cluster cards ---
    st.subheader("Customer Segments — Response Typology")
    st.caption(
        "Customers were clustered by their estimated causal treatment effect (CATE). "
        "Four distinct groups emerged, each with a different recommended action."
    )

    cols = st.columns(4)
    cluster_order = [0, 1, 2, 3]
    for col, cid in zip(cols, cluster_order):
        row = cluster_profiles[cluster_profiles["cate_cluster"] == cid].iloc[0]
        color = CLUSTER_COLORS[cid]
        action = CLUSTER_ACTIONS[cid]
        n = int(row["n"])
        cate = row["cate_any"]
        cate_str = f"${cate:+.2f}"
        conv_str = f"{row['actual_conv']:.1%}"
        total_rev = n * cate
        if cid == 0:
            # C0: frame as waste avoided by suppressing
            total_line = (
                f"<span style='color:{color};font-weight:600;'>"
                f"Suppressing saves ~${abs(total_rev):,.0f}</span>"
            )
        else:
            total_line = (
                f"<span style='color:{color};font-weight:600;'>"
                f"Est. total revenue: ~${total_rev:,.0f}</span>"
            )
        with col:
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding: 0.75rem 1rem; "
                f"border-radius: 0.5rem; background: #f8fafc; margin-bottom: 0.5rem;'>"
                f"<div style='font-size:0.75rem; font-weight:600; color:{color}; "
                f"text-transform:uppercase; letter-spacing:0.05em;'>Cluster C{cid}</div>"
                f"<div style='font-size:1.05rem; font-weight:700; margin: 0.25rem 0;'>"
                f"{row['persona']}</div>"
                f"<div style='font-size:0.85rem; color:#64748b;'>"
                f"n = {n:,} customers<br>"
                f"Mean CATE: <strong>{cate_str}</strong><br>"
                f"Conversion rate: {conv_str}<br>"
                f"Avg history: ${row['mean_history']:.0f}<br>"
                f"{total_line}</div>"
                f"<div style='margin-top:0.5rem; font-size:0.8rem; font-weight:600; "
                f"color:{color};'>ACTION: {action}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.plotly_chart(make_cluster_cate_bar(cluster_profiles), width="stretch")

    st.divider()

    # --- Policy recommendation ---
    st.subheader("Email Policy: Two Deployment Tiers")
    st.caption(
        "Two versions of the policy are available depending on whether a scoring pipeline "
        "can be run at send time."
    )

    tab_policy1, tab_policy2 = st.tabs(["Comparison", "Send Breakdown"])

    with tab_policy1:
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(make_policy_comparison_chart(policy), width="stretch")
        with right:
            st.markdown("**Tier 1 — Rule-Only Policy**")
            st.markdown(
                "Suppresses ~2,127 customers via interpretable if/then rules derived "
                "from the policy tree. No scoring pipeline needed — rules can be applied "
                "in a spreadsheet or CRM filter.\n\n"
                "61,873 customers emailed | $0.70 mean CATE | **~$43,300 total est. revenue**"
            )
            st.markdown("---")
            st.markdown("**Tier 2 — C0-Filtered Policy (Recommended)**")
            st.markdown(
                "Applies the CATE cluster model as an upstream gate: if a customer's "
                "estimated uplift is negative (Cluster C0), suppress regardless of email type. "
                "Suppresses **16,889 customers** — the full organic-buyer segment.\n\n"
                "47,111 customers emailed | $1.09 mean CATE | **~$51,400 total est. revenue**\n\n"
                "That's **+$11,100 vs a blanket send** — from sending to fewer people."
            )

    with tab_policy2:
        st.plotly_chart(make_send_breakdown_chart(policy), width="stretch")
        policy_counts = policy["policy_c0_filtered"].value_counts().reset_index()
        policy_counts.columns = ["Recommendation", "Customers"]
        policy_counts["Share"] = (policy_counts["Customers"] / 64000 * 100).round(1).astype(str) + "%"
        st.dataframe(policy_counts, hide_index=True, width="stretch")

    st.divider()

    # --- Suppression gap explainer ---
    st.subheader("The Suppression Gap: Why 16,889 Customers, Not 2,127?")

    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            """
The model finds two layers of suppression:

**Layer 1 — Rule-based (~2,127 customers)**
The policy tree identifies a sharp feature boundary: customers with
*history ≤ $30, lapsed > 6.5 months, and established (not a newbie)*
have reliably negative uplift and can be expressed as a simple rule.

**Layer 2 — Score-based (~14,762 more customers)**
Cluster C0 contains 16,889 customers total with negative predicted CATE.
Most of them don't sit at a single clean feature boundary — their
suppression requires the continuous CATE score from Stage 3.

**This is not a failure of the rule-based approach.** Rules can only
express boundaries in feature space. The continuous score expresses
boundaries in response space. Both are valid tools for different
deployment contexts.
            """
        )
    with right:
        gap_fig = go.Figure(go.Bar(
            x=["Blanket send\n(no model)", "Tier 1: Rules only\n(no infra)", "Tier 2: C0 filter\n(scoring pipeline)"],
            y=[64000, 61873, 47111],
            text=["64,000 sent\n$0.63 CATE", "61,873 sent\n$0.70 CATE", "47,111 sent\n$1.09 CATE"],
            textposition="outside",
            marker_color=["#94a3b8", "#60a5fa", "#22c55e"],
        ))
        gap_fig.update_layout(
            title="Customers Emailed by Policy Tier",
            yaxis_title="Customers emailed",
            yaxis=dict(range=[0, 76000]),
            height=380,
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(gap_fig, width="stretch")

    st.divider()

    # --- Recency x Newbie heatmap ---
    st.subheader("Key Insight: Recency x New Customer Status")
    st.caption(
        "Not all lapsed customers are the same. Splitting by new-customer status "
        "reveals fundamentally different email responses. This interaction is the "
        "strongest segmentation signal in the dataset."
    )

    st.plotly_chart(make_recency_newbie_heatmap(cdf, recency_labels), width="stretch")

    insight_cols = st.columns(3)
    insight_cols[0].info(
        "**Lapsed newbies = Mens win-back**\n\n"
        "Customers lapsed 10-12 months who are still within their first year "
        "show +$1.26 Mens spend lift — the highest segment in the dataset."
    )
    insight_cols[1].warning(
        "**Established customers: skip Womens**\n\n"
        "Customers who joined before the last 12 months show near-zero or "
        "negative response to Womens email across all recency levels."
    )
    insight_cols[2].success(
        "**High-priority micro-segment**\n\n"
        "Newbie + both catalogs + Multichannel (n=1,372): "
        r"Mens +\$1.89, Womens +\$2.04. Use RCT averages — model is data-sparse here."
    )


# ---------------------------------------------------------------------------
# TECHNICAL VIEW
# ---------------------------------------------------------------------------
else:
    st.title("Technical Details: Causal ML Pipeline")
    st.caption(
        "Hillstrom e-mail marketing dataset — 64,000 customers, 3-arm RCT "
        "(Mens Email / Womens Email / No Email control). "
        "Pipeline: T-Learner (conversion) + Causal Forest (spend) + Policy Tree (deployment rules)."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "CATE Analysis", "Policy Tree Rules", "Feature Importances", "Methodology & Limits"
    ])

    # --- Tab 1: CATE Analysis ---
    with tab1:
        st.subheader("CATE Score Distributions")
        st.caption(
            "CATE = Conditional Average Treatment Effect. Each customer receives a "
            "personalized estimate of incremental spend from email. "
            "Negative CATE = emailing this customer is expected to reduce net revenue."
        )
        st.plotly_chart(make_cate_density(scores), width="stretch")
        st.caption(
            "Dashed lines show mean CATE per email type — "
            r"Any Email: **\$0.63** | Mens Email: **\$0.72** | Womens Email: **\$0.46**"
        )

        st.markdown("**Mean CATE by email type:**")
        ate_data = {
            "Email Type": ["Any Email", "Mens Email", "Womens Email"],
            "Mean CATE": ["$0.63", "$0.72", "$0.46"],
            "% Customers Positive CATE": ["76.5%", "72.1%", "63.6%"],
            "Naive ATE (raw diff)": ["$0.60", "$0.77", "$0.42"],
        }
        st.dataframe(pd.DataFrame(ate_data), hide_index=True, width="stretch")

        st.subheader("CATE by Quartile — Actual vs Predicted")
        st.caption(
            "Validation: customers sorted into quartiles by predicted CATE, "
            "then actual spend lift measured. Q4 (top quartile) captures 87% of actual "
            "spend lift while representing only 25% of customers."
        )
        quartile_data = {
            "Quartile": ["Q1 Low", "Q2", "Q3", "Q4 High"],
            "n": ["16,000", "16,004", "15,996", "16,000"],
            "Mean CATE (predicted)": ["-$1.19", "+$0.20", "+$0.75", "+$2.77"],
            "Actual spend lift": ["-$2.34", "+$0.01", "+$0.24", "+$4.49"],
        }
        st.dataframe(pd.DataFrame(quartile_data), hide_index=True, width="stretch")

        st.subheader("CATE Cluster Profiles")
        display_cols = [
            "cate_cluster", "persona", "n", "cate_any", "cate_mens", "cate_womens",
            "mean_history", "mean_recency", "pct_newbie", "actual_conv",
        ]
        display_df = cluster_profiles[display_cols].copy()
        display_df.columns = [
            "Cluster", "Persona", "n", "CATE Any", "CATE Mens", "CATE Womens",
            "Avg History ($)", "Avg Recency (mo)", "% Newbie", "Actual Conv",
        ]
        for col in ["CATE Any", "CATE Mens", "CATE Womens"]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        display_df["Avg History ($)"] = display_df["Avg History ($)"].apply(lambda x: f"${x:.0f}")
        display_df["Avg Recency (mo)"] = display_df["Avg Recency (mo)"].apply(lambda x: f"{x:.1f}")
        display_df["% Newbie"] = display_df["% Newbie"].apply(lambda x: f"{x:.0%}")
        display_df["Actual Conv"] = display_df["Actual Conv"].apply(lambda x: f"{x:.2%}")
        st.dataframe(display_df, hide_index=True, width="stretch")

    # --- Tab 2: Policy Tree Rules ---
    with tab2:
        st.subheader("Policy Tree Rules (Untuned, Depth=3)")
        st.caption(
            "Shallow decision tree trained to maximize expected incremental revenue. "
            "These rules are human-readable and deployable without a scoring pipeline."
        )
        st.code(rules_text, language=None)

        st.subheader("Tuning Results Summary")
        st.caption(
            "5-fold cross-validation across 27 hyperparameter combinations. "
            "Any min_impurity_decrease > 0 collapses trees to depth=0 (send everyone). "
            "Conservative lower-bound target sends 0% — CIs too wide to act on."
        )
        tuning_data = {
            "Tree": ["Any Email", "Mens Email", "Womens Email"],
            "Best depth": [4, 4, 3],
            "Best min_leaf": [500, 200, 200],
            "CV policy value": ["$0.666", "$0.754", "$0.659"],
            "Customers sent (tuned)": ["97.4%", "93.6%", "84.8%"],
            "Rule-based DNE": [0, 0, 0],
        }
        st.dataframe(pd.DataFrame(tuning_data), hide_index=True, width="stretch")

        st.info(
            "**Why does the tuned tree find 0 rule-based suppressions?**\n\n"
            "The depth=3 untuned tree finds ~2,127 suppressions, but this split does not "
            "stabilize across CV folds — it fires in some folds and not others. "
            "The boundary (history $\\leq$ $30, lapsed, established) is real but fragile "
            "at the partition level. The C0 cluster filter is more reliable for suppression "
            "because it operates on continuous CATE scores rather than feature thresholds."
        )

    # --- Tab 3: Feature Importances ---
    with tab3:
        st.subheader("Feature Importances — Causal Forest (Stage 3)")
        html_content = load_html("outputs/03_feature_importance.html")
        st.iframe(html_content, height=500)

        st.subheader("Feature Importances — Baseline XGBoost (Stage 1)")
        html_content_01 = load_html("outputs/01_feature_importance.html")
        st.iframe(html_content_01, height=500)

        st.markdown(
            """
**Feature engineering notes (v2 pipeline):**

- `log_history` — raw history is right-skewed ($29.99 floor). Log-transform used throughout.
- `zip_Suburban`, `zip_Urban` — one-hot encoded; Rural is reference (weakest responder).
  Label-encoding imposed false ordinal structure that masked the Suburban/Urban email-type flip.
- `channel_Multichannel`, `channel_Web` — one-hot; Phone is reference (weakest responder).
- `recency_x_newbie` — explicit multiplicative interaction. Lapsed newbies respond to Mens
  email at 2x the rate of lapsed established customers. Without this term, the forest
  sees recency and newbie marginally but cannot detect their joint effect directly.
- `both_catalogs` — customers who purchased from both catalog types (n=6,448).
  Within-group EV discrimination is flat (Q1=\$1.81 vs Q4=\$1.95) — data sparsity problem.
  Use RCT group averages for this segment rather than model scores.
            """
        )

    # --- Tab 4: Methodology & Limitations ---
    with tab4:
        st.subheader("Pipeline Overview")
        pipeline_data = {
            "Stage": ["1", "2", "3", "4", "5", "6", "6b", "7", "7b"],
            "Component": [
                "Baseline XGBoost",
                "T-Learner (Conversion)",
                "Causal Forest (Spend)",
                "Expected Value Ranking",
                "Feature Clustering",
                "CATE Clustering",
                "Recency/Newbie Exploration",
                "Policy Tree",
                "Policy Tree Tuning",
            ],
            "Purpose": [
                "Feature importance baseline, data sanity check",
                "P(converts | email) - P(converts | no email) per customer",
                "E[spend | email] - E[spend | no email] — the primary CATE estimate",
                "Combine Stages 2+3 into EV score; produce ranked send list",
                "K-means on customer attributes; natural segment structure",
                "K-means on CATE vectors; response typology (suppress / high-value / moderate)",
                "RCT cross-tab: recency x newbie x email type — causal estimates within cells",
                "Econml PolicyTree — interpretable if/then rules maximizing expected revenue",
                "5-fold CV across 27 param combos; C0 upstream filter for suppression",
            ],
        }
        st.dataframe(pd.DataFrame(pipeline_data), hide_index=True, width="stretch")

        st.subheader("Why Causal ML, Not Standard Uplift?")
        st.markdown(
            """
Standard conversion-rate models answer: *who is likely to buy?*
Causal uplift models answer: *who buys **because of** the email?*

The difference matters:
- A customer with 5% baseline conversion who converts at 5.1% after email generates
  almost no incremental revenue — the email is wasted.
- A customer with 0.5% baseline who converts at 3.0% after email is a strong target
  even though their absolute conversion rate looks low.

The Causal Forest (econml) directly estimates this difference using the
Generalized Random Forest framework, which is robust to confounding within
randomized treatment assignment.

The T-Learner trains separate XGBoost models on treated and control subsets,
then subtracts predictions to estimate uplift. This is simpler but subject to
confounding from feature interactions between model and treatment selection.
Both are used here: T-Learner for conversion probability, Causal Forest for spend.
            """
        )

        st.subheader("Limitations")
        st.markdown(
            """
1. **Spend-given-conversion instability:** Only 578 converters total (0.9% rate),
   split across 3 treatment arms (~122-456 per arm). The spend-given-conversion
   sub-models in Stage 4 have high variance. CATE estimates for conversion
   (Stage 2) are more reliable than spend estimates (Stage 3).

2. **Both-catalog sparsity:** 6,448 customers purchased from both catalog types.
   Within-group EV discrimination is essentially flat (Q1=\$1.81 vs Q4=\$1.95).
   This is a sample-size constraint — ~1,300 per treatment arm is insufficient
   for the forest to detect within-group heterogeneity. Use the RCT group average
   (\$1.45 Mens, \$2.04 Womens for Multichannel+newbie+both) not the model score.

3. **Micro-segment sample sizes:** Segments like Rural+Multichannel+newbie
   have too few observations for reliable CATE estimates. Point estimates exist
   but confidence intervals are wide.

4. **Single time-period RCT:** The 2008 Hillstrom dataset represents one campaign
   snapshot. Customer behavior and channel responsiveness change over time.
   The model should be retrained before applying to a new campaign.

5. **Policy tree suppression vs CATE cluster suppression:** The rule-based
   suppress boundary (history ≤ $30, lapsed, established) is real but does not
   stabilize across CV folds. The C0 cluster filter is more reliable but requires
   the scoring pipeline. For a no-infra deployment, the rule-based approach is a
   defensible starting point that captures the sharpest suppressions.
            """
        )
