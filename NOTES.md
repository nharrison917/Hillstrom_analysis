# Hillstrom Uplift Modeling — Project Notes

Design decisions, analytical findings, and reasoning behind each stage.
For portfolio, interview, and report reference.

---

## Why This Project Matters

Standard email targeting asks: "who is likely to buy?" That selects customers who would
have bought anyway — the email is wasted on them. Uplift modeling asks the harder question:
"who buys *because of* the email?" This is the causal question, and it changes who you target.

The Hillstrom dataset is a randomized controlled trial (RCT), which means we can actually
estimate causal effects cleanly — unlike most real-world marketing data where treated and
untreated customers differ systematically.

---

## Stage-by-Stage Reasoning

### Stage 1 — Baseline XGBoost (ROC-AUC 0.55)
Intentionally weak. We excluded the treatment indicator from features.
Why: we wanted to confirm that observable customer features alone don't predict conversion
well. If baseline AUC were high, it would mean we could target by propensity and get most
of the lift. The low AUC is evidence that *causal* modeling is necessary — the "who would
have bought anyway" problem is real here.

### Stage 2 — T-Learner Conversion Uplift
Fits two models: P(convert | email) and P(convert | no email). Uplift = difference.
Limitation: T-Learner is susceptible to noise because two separate models can drift apart
even when the true difference is small. Works best when outcomes are frequent enough to
estimate each surface reliably. With 0.9% conversion, this is marginal but usable.

### Stage 3 — Causal Forest on Spend
CausalForestDML from econml. More honest than T-Learner: it uses doubly-robust estimation,
meaning it's protected against misspecification in either the outcome model or the treatment
model. Produces CATE (conditional average treatment effect) per customer plus confidence intervals.

Key limitation: only 578 converters (0.9%) means the spend surface is very sparse.
The causal forest picks up signal but confidence intervals are very wide — especially for
the lower bound, which averages -$1.63. The point estimates are our best guess; the CIs
tell us how uncertain that guess is.

Three comparisons run: any email, mens email, womens email vs. no email control.
Mens and Womens CATE were point estimates only (CI bounds not computed for those).

### Stage 4 — Expected Value Ranking
Combined conversion probability (Stage 2) and revenue lift (Stage 3) into a single
expected value per customer. This is the continuous scoring layer — the rank-ordered
send list. Used as the reference comparison for all downstream rule-based work.

### Stage 5 — Feature Clustering (k=3)
K-means on customer features. Found that the dominant segmentation is catalog type
(mens vs womens vs both), not channel or geography. This shapes the email type
recommendation but doesn't directly drive suppression decisions.

### Stage 6 — CATE Clustering (k=4)
K-means on CATE estimates (revenue lift scores). This is the most important segmentation.

| Cluster | n      | Persona                | Mean CATE | Decision      |
|---------|--------|------------------------|-----------|---------------|
| C0      | 17,233 | Organic Buyers         | -$0.59    | Suppress      |
| C1      | 4,703  | Moderate (conversion)  | +$0.57    | Send either   |
| C2      | 38,588 | Moderate (broad)       | +$0.75    | Send Mens     |
| C3      | 3,476  | High Value Targets     | +$5.58    | Priority send |

C0 is the critical finding: 17,233 customers generate *negative* incremental revenue
from email. They would have bought anyway, and the email may even suppress spend
(fatigue, framing effects). Emailing them is a cost with no benefit.

C3 profile: mean history $396, Urban skew, 9%+ actual conversion rate vs 0.09% control.
The lift is real and large — this is who you protect above all else.

### Stage 7 — Policy Tree (initial)
Shallow decision tree fitted to maximize expected CATE across the feature space.
Output is a small set of if/then rules deployable without any scoring infrastructure.

Initial parameters: depth=3, min_samples_leaf=500.
Result: 2,127 customers cleanly suppressed via rule.

The gap between 2,127 (rules) and 17,233 (C0 cluster) was the open question going into 7b.

---

## Stage 7b — Policy Tree Tuning: Full Analysis

### What We Changed and Why

**1. 5-fold cross-validation instead of a single train/test split**

A single 80/20 split gives one estimate of policy value. Depending on which customers
land in the test set, that estimate varies. With k-fold we run 5 different splits and
average the results, so the hyperparameter selection is stable rather than lucky.

We stratified on the treatment group (Mens Email / Womens Email / No Email) so each fold
has a proportional mix — important because the CATE estimates are derived from treatment
contrasts and an imbalanced fold would distort them.

**2. min_impurity_decrease sweep (pre-pruning)**

A node only splits if the gain in mean policy value (mean CATE among recommended-send
customers in that node) exceeds this threshold. We swept [0.0, 0.01, 0.05].

Purpose: prevent the tree from chasing tiny gains in the training data that won't
generalize. This is a regularization mechanism — the tree has to "earn" each split.

**3. Conservative training target (lower confidence bound)**

Instead of training on the causal forest's CATE point estimate, we tested training on the
5th-percentile lower bound (cate_any_lb). The idea: only target customers where even the
*conservative* estimate of lift is positive. This filters customers whose positive CATE
might just be estimation noise rather than real signal.

Only available for the "any email" tree — Stage 3 computed CI bounds for the any-email
comparison but not individually for mens/womens.

---

### Key Findings from Tuning

**Finding 1: The conservative lower-bound approach suppressed everyone (0% sent)**

Mean cate_any_lb = -$1.63. 59,006 of 64,000 customers have negative lower bounds.

What this tells us: the causal forest's confidence intervals are extremely wide. With
578 converters in 64,000 rows, the model cannot be statistically confident that any
specific customer group has positive CATE. The point estimates are real but uncertain.

This is NOT a failure of the model — it's the model being honest about what it knows.
The correct interpretation: "we believe these customers will respond, but the signal is
noisy because conversions are rare." The point estimate is still the best available
targeting signal; the wide CI is context, not a veto.

Interview framing: "I tested a conservative targeting approach and found the confidence
intervals were too wide to support statistically certain positive-CATE identification.
This tells you something important about the data — 578 converters is near the floor for
reliable causal estimation at this granularity. The targeting still works on expectation;
you just can't guarantee it for any individual customer."

**Finding 2: Any min_impurity_decrease > 0 collapses all trees**

Setting min_impurity_decrease to 0.01 or 0.05 caused every tree to make zero splits —
all customers got the same recommendation (100% sent). The reason: every possible split
in the policy tree improves mean CATE by less than $0.01. The CATE surface is very flat
across feature space.

What this means: the differences in CATE across feature groups are real but small. The
policy tree is making marginal distinctions, not sharp separations. Pre-pruning at any
positive threshold removes all those marginal splits, leaving nothing.

Implication for the business case: the value of the policy tree is not that it finds
dramatically different customer segments — it's that it finds the small subset where the
CATE is clearly and reliably negative, worth suppressing even without a large effect size.

**Finding 3: The suppression gap is unchanged — and now explained**

Stage 7b DNE: 2,127. Stage 7 DNE: 2,127. Tuning had no effect.

This is the correct result. Here is why:

The C0 cluster (17,233 customers) is defined by CATE *scores* — the output of the
causal forest. Those scores reflect subtle patterns that the forest can detect across
many trees. A policy tree operates in *feature space* — it can only say "history <= $X
AND channel = Phone AND recency > Y". The C0 customers don't all share a clean
feature-space boundary. They're scattered across history values, channels, and
recencies that partly overlap with positive-CATE customers.

The 2,127 with clean rules sit at a sharp, non-overlapping boundary: history = $29.99
(the lowest history segment), older customers (not newbie), longer-lapsed (recency > 6-7
months). That boundary is so clean that even a depth-2 tree finds it. The other ~15,000
C0 customers don't have a comparable boundary at depth <= 4.

The right mental model: a policy tree is a coarse filter. A causal forest score is a
fine-grained filter. Both have a role. The clean rules are deployable without a model;
the scores are better for customers where the evidence is mixed.

**Finding 4: Mens tree collapsed to actual depth=2 despite requesting depth=4**

The tree was fitted with max_depth=4 but the fitted depth was 2. This means the tree
couldn't find any beneficial split beyond what depth=2 already captured. It's a clean
validation: depth=2 is the natural complexity of the Mens suppression problem.

**Finding 5: Womens tree showed high fold variance (std=0.11) at depth 3-4**

Some CV folds found a suppression leaf; others didn't. This is the definition of
overfitting to the CATE surface — the tree is finding patterns in the noise, not
stable patterns in the signal. When the training set changes slightly (one fold vs.
another), the result flips.

The practical implication: the Womens tree's suppression rules should be treated with
more caution than the Mens rules. The one suppression leaf it does find consistently
(Phone/Web, history <= $30, recency > 6.5, not newbie) is the stable part.

**Finding 6: Leakage — 292 customers suppressed for Mens but wrongly sent Womens**

1,906 customers in the Mens-suppress leaf are still being sent Womens email.
Of those, 292 also have negative Womens CATE. These are customers who should be
suppressed entirely but aren't, because the Mens and Womens trees are fitted independently.

Fix (not yet implemented): use the CATE C0 cluster as an upstream gate. If a customer
is in C0, suppress regardless of what the individual email-type trees say. This would
catch the 292 cleanly.

---

## The Two-Layer Recommendation Architecture

This project produces two complementary outputs:

**Layer 1 — Rule-based suppression (Stage 7b policy tree)**
- 2,127 customers: Do Not Email, for both email types
- Rules are simple, auditable, and deployable without a scoring system
- Profile: history=$29.99, Phone/Web channel, recency > 6-7 months, not a newbie
- Best for: compliance review, list management, operations teams

**Layer 2 — Score-based targeting (Stage 4 send list)**
- Continuous CATE scores for all 64,000 customers
- Suppresses the full ~17,233 C0 customers by selecting only those with positive EV
- Best for: campaign systems that can ingest a scored list, A/B testing, budget-constrained sends

The gap between the two is not a failure — it's expected. Rules are for clean boundaries;
scores are for ambiguous cases. A real deployment would use both: suppress the rule-based
2,127 first (no model needed), then apply the score-based filter to the rest.

---

## Stage 6b Findings — Recency and Newbie Interactions

### What we found

**The key interaction is recency x newbie, not recency alone.**

Recency by itself doesn't identify suppressable customers — all four recency buckets
show positive Mens email lift. But once you split by newbie status, a clear pattern
emerges:

| Segment | Mens lift | Womens lift |
|---------|-----------|-------------|
| Recent (1-2mo) x Established | +$0.83 | **-$0.21** |
| Recent (1-2mo) x Newbie | +$1.00 | +$1.15 |
| Lapsing (7-9mo) x Established | +$0.37 | **-$0.14** |
| Lapsing (7-9mo) x Newbie | +$0.86 | +$0.78 |
| Lapsed (10-12mo) x Newbie | **+$1.26** | +$0.57 |
| Lapsed (10-12mo) x Established | +$0.34 | +$0.28 |

**Three actionable findings:**

1. **Established customers barely respond to Womens email overall** (+$0.11 spend lift
   vs +$0.73 for newbies). For the Womens email specifically, newbie status is the
   dominant moderator of response.

2. **Recent established customers have NEGATIVE Womens email lift (-$0.21).** These
   are customers who bought recently, already satisfied — the Womens email doesn't
   move them. This is consistent with the "organic rebuyer" hypothesis: they were
   going to engage anyway, and the email may displace natural spend timing.

3. **Lapsed newbies (+$1.26 Mens lift) are the highest-responding segment.**
   Counterintuitive: very lapsed (10-12 months) but newly acquired. Interpretation:
   they were acquired, went quiet, and the email re-activates them. This is a classic
   win-back scenario that email is designed for.

**High-value micro-segment confirmed:**
Newbie + both_catalogs + Multichannel (n=1,372): Mens lift +$1.89, Womens lift +$2.04.
These are the most email-responsive customers in the dataset. Too small for the causal
forest to discriminate within (item #3 in POSSIBLE_DIRECTIONS.md), but confirmed as
a priority target via cross-tab.

### What this means for the model

The current causal forest has `newbie` as a feature but no recency x newbie interaction
term. The model likely averages across these groups rather than capturing the flip in
Womens email response between established and newbie customers.

For the report: these findings from the RCT cross-tab are valid causal estimates
(randomisation holds within each segment). They can be cited directly as empirical
evidence, not model predictions.

---

## C0 Upstream Filter (Stage 7b addition)

Applying the CATE cluster C0 assignment as an upstream suppression gate produces:

| Policy | DNE | % Suppressed | Policy value (sent customers) |
|--------|-----|-------------|-------------------------------|
| Rule-only (policy tree) | 2,127 | 3.3% | $0.71 |
| C0-filtered | 17,831 | 27.9% | **$1.10** |

The C0 filter raises the mean CATE of sent customers by 56% ($0.71 → $1.10). This is
the quantified value of suppressing the organic buyer segment. It answers the business
question: "what do we gain by not emailing people who were going to buy anyway?"

Trade-off: the C0 filter requires the Stage 3/6 scoring pipeline. The rule-only list
requires no model infrastructure — just the if/then conditions from the policy tree.
Both are valid; the right choice depends on deployment context.

---

## Known Limitations (for report and interviews)

1. **Sparse conversions (578/64,000 = 0.9%)**: The causal forest is estimating a spend
   surface with very few signal events. CI bounds are wide. Point estimates are directionally
   correct but should be treated as population-level guidance, not individual precision.

2. **Multichannel lift not captured**: Multichannel customers show ~2x the lift of single-
   channel customers, but the model doesn't have an interaction feature for channel x email type.
   Adding channel:email_type interaction features would likely tighten the Multichannel suppression.

3. **Both-catalog segment (2.5x lift)**: Customers with mens=1 AND womens=1 show
   disproportionate lift, but the group is too small (6,448 customers) for the causal forest
   to carve out a reliable sub-estimate.

4. **Zip code x email type interactions not captured**: Suburban responds to Mens, Urban to
   Womens. The model averages across zip codes within email type. A zip x email_type feature
   might reveal cleaner targeting.

5. **Policy tree is a point estimate tool**: The policy tree trains on CATE point estimates,
   not on actual outcomes. If the causal forest overestimates lift in some regions (noisy CATE
   surface), the policy tree will target those regions. The CV policy value we report is still
   mean estimated CATE, not observed spend. Ground truth validation would require a new experiment.

6. **Suppression leakage (Blind Spot #7)**: 292 customers with negative CATE for both email
   types are still in the send list because the Mens and Womens trees are fitted independently.
   A C0 upstream filter would eliminate this.

---

## What This Demonstrates (Portfolio Framing)

- **Causal thinking**: The project is built around the right question (incremental effect),
  not the easy question (who converts). This is the distinction between uplift modeling and
  classification.

- **Honest model evaluation**: CI bounds, fold variance, suppression gap analysis — the work
  doesn't just report the headline number. It characterizes what the model can and can't do.

- **Layered recommendation system**: Rules for clean cases, scores for ambiguous ones.
  This is how real production systems work.

- **Business-deployable output**: The 2,127-customer suppression list is a direct operational
  artifact. No scoring pipeline required. A manager can implement it in a spreadsheet.

- **Documented limitations**: Every model has failure modes. Documenting them is a sign of
  rigor, not weakness.
