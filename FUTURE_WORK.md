# Possible Directions / Model Blind Spots

Findings from exploratory analysis that the current models do not fully capture.
Each section describes what was observed, why the model misses it, and what a
follow-up approach could look like.

---

## 1. Channel x Email Responsiveness

**What we found:**
Multichannel customers convert at 1.26% overall vs 0.77-0.93% for Phone/Web.
They also carry 2.5x the prior spend history and are 3x more likely to have
bought from both catalogs. Email lift is strongest for this group:
- Multichannel Mens lift: +1.02pp
- Web Mens lift: +0.72pp
- Phone Mens lift: +0.55pp

Phone customers show almost no response to the Womens email (+0.17pp, +$0.23 spend).

**Why the model misses it:**
`channel` is one label-encoded integer among nine features. The causal forest
sees it as a weak marginal predictor rather than a strong moderator of treatment
response. The EV quartile distribution is flat across all three channels (~25% each).

**Possible direction:**
- Fit channel-specific causal forest models (separate model per channel)
- Add explicit interaction features: `channel_x_mens`, `channel_x_womens`
- Use a policy tree / uplift tree that splits on channel early

---

## 2. Zip Code x Email Type Interaction

**What we found:**
Suburban customers respond well to Mens email (+0.74pp lift, highest of any zip)
but barely respond to Womens email (+$0.18 spend lift, nearly flat).
Urban customers show the opposite pattern: Womens email drives +$0.67 spend lift,
almost matching Mens.
Rural customers convert at the highest base rate (1.12%) despite similar prior
history to other zip codes -- likely reflecting catalog-channel affinity.

**Why the model misses it:**
Same issue as channel -- `zip_code` is a single encoded feature. The model
captures average spend differences but not differential email type responsiveness
by geography.

**Possible direction:**
- Separate Mens vs Womens email models per zip code
- Add `zip_x_email_type` interaction features
- Investigate Rural catalog affinity further -- may warrant its own segment model

---

## 3. Both-Catalog Customers (Mens=1 and Womens=1)

**What we found:**
6,448 customers (10.1%) have purchased from both catalogs. This group:
- Converts at 2.43% when sent Mens email vs 0.93% for single-catalog customers
- Shows 2.5x the Mens email lift of single-catalog buyers (+1.50pp vs +0.59pp)
- Has a "Do Not Email" subgroup (22%, n=1,429) that converts at 1.96% organically
  -- these are genuine organic buyers who don't need the nudge

**Why the model misses it:**
The `both_catalogs` flag is a binary feature but there are only 6,448 observations.
The causal forest cannot reliably estimate heterogeneous effects within this subgroup
at that sample size. EV top/bottom quartile within this group shows almost no
difference in actual spend ($1.81 vs $1.95), confirming the model is not
discriminating well inside this segment.

**Possible direction:**
- Train a dedicated uplift model on both-catalog customers only
- Treat both-catalog as a priority override: if both_catalogs=1, send Mens email
  unless Do Not Email flag is set by a separate suppression model
- Investigate whether both-catalog customers who *also* use Multichannel channel
  represent a reliably identifiable high-value micro-segment

---

## 4. Channel x Zip Interaction (Micro-segments)

**What we found:**
Rural Multichannel customers have the highest observed conversion rate of any
channel/zip combination (1.42%). No single feature captures this -- it only
emerges from the cross-tab.

**Why the model misses it:**
No interaction features between channel and zip_code exist in the current feature
set. Both are treated as independent marginal predictors.

**Possible direction:**
- Add `channel_zip` combination as a single categorical feature
- Use a tree-based segmentation (e.g. policy tree from econml) that can discover
  these combinations as natural splits
- Check if Rural Multichannel + both_catalogs is a stable high-value micro-segment
  worth targeting as a named persona

---

## 5. Spend-Given-Conversion Model Instability

**What we found:**
The Stage 4 EV decomposition fits separate spend-given-conversion models on
converters only. Sample sizes are small:
- Any treated converters: 456
- Control converters: 122
- Mens email converters: 267
- Womens email converters: 189

The top/bottom EV quartile within the both-catalog group shows inverted actual
spend ($1.81 top vs $1.95 bottom), suggesting the spend-given-conversion model
is noisy in small subgroups.

**Possible direction:**
- Use the causal forest CATE directly (Stage 3) as the primary ranking signal
  rather than the decomposed EV -- it is more stable
- Pool treated groups when estimating spend-given-conversion to increase sample
- Consider a hurdle model (logistic for conversion + regression for spend) with
  regularization to stabilize small-sample estimates

---

## 6. Recency and Newbie Interactions

**EXPLORED in Stage 6b (`06b_recency_newbie.py`). Key findings:**

The key signal is the recency x newbie INTERACTION, not either feature alone.

- **Established customers respond poorly to Womens email** (+$0.11 overall).
  Recent established (1-2mo) have NEGATIVE Womens lift (-$0.21). Consistent with
  the organic rebuyer hypothesis: recently satisfied customers don't need a nudge.
- **Newbies respond well to both email types** across all recency buckets.
  Lapsed newbies (10-12mo) have the highest Mens lift in the dataset (+$1.26).
  Interpretation: acquired but gone quiet; email is a true re-activation tool here.
- **Newbie + both_catalogs + Multichannel** (n=1,372): Mens +$1.89, Womens +$2.04.
  Highest-responding micro-segment. Too small for causal forest to discriminate,
  but confirmed as priority target via RCT cross-tab.

The causal forest has `newbie` as a feature but no recency x newbie interaction term.
Adding this interaction would likely improve Womens email suppression for established
customers. See NOTES.md for full breakdown table.

**Outputs:** `outputs/06b_recency_lift.html`, `outputs/06b_recency_newbie_heatmap.html`,
`outputs/06b_newbie_lift.html`

---

## 7. Policy Tree Suppression Leakage

**RESOLVED in Stage 7b (`07b_policy_tree_tuning.py`).**

After tuning, leakage is 292 customers (Mens-suppress, sent Womens, negative Womens CATE).
Down from 323 in Stage 7 due to updated tree parameters.

**Fix implemented:** C0 upstream suppression filter.
If `cate_cluster == 0` (Stage 6 CATE clustering), override recommendation to Do Not Email
regardless of individual tree decisions. Results:

- Rule-only: 2,127 DNE, mean policy value of sent customers = $0.71/customer
- C0-filtered: 17,831 DNE, mean policy value of sent customers = $1.10/customer

The filter raises targeting quality by 56%. It requires the Stage 3/6 scoring pipeline
to deploy; the rule-only version requires no model infrastructure. Both are saved in
`outputs/policy_tree_recommendations.csv` (columns: `policy_recommendation`,
`policy_c0_filtered`).

---

## Summary Table

| Blind Spot | Impact | Complexity to Fix | Priority |
|---|---|---|---|
| Channel x email responsiveness | High (2x lift difference) | Medium | High |
| Both-catalog segment model | High (2.5x lift, model can't see it) | Medium | High |
| Zip x email type interaction | Medium (Suburban/Urban flip) | Low-Medium | Medium |
| Channel x Zip micro-segments | Medium | Medium | Medium |
| Spend-given-conversion instability | Low-Medium (affects EV decomp) | Low | Medium |
| Recency/Newbie interactions | **EXPLORED** -- see item 6 | Done | Done |
| Policy tree suppression leakage | **RESOLVED** -- C0 filter implemented | Done | Done |
