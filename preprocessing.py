# -*- coding: utf-8 -*-
"""
preprocessing.py -- Shared feature engineering for all Hillstrom pipeline stages.

Import this instead of repeating feature engineering in each script:

    from preprocessing import prepare_features, FEATURES, RANDOM_STATE, DATA_FILE
    df = pd.read_csv(DATA_FILE)
    df = prepare_features(df)
    X = df[FEATURES]

Feature set changes from v1 (original pipeline):
  - zip_code:  was label-encoded (1 ordinal int, false ordering imposed)
               now one-hot: zip_Suburban, zip_Urban  (Rural = reference category)
  - channel:   was label-encoded (1 ordinal int, false ordering imposed)
               now one-hot: channel_Multichannel, channel_Web  (Phone = reference)
  - recency_x_newbie: new explicit interaction term
  - history_segment:  kept as ordinal label-encoding (genuine spend-tier ordering)

Reference categories chosen as the weakest email responders:
  - Phone:  lowest email lift across Mens and Womens
  - Rural:  lowest observed conversion lift; catalog affinity may explain base rate

With one-hot encoding, policy tree splits name the interesting categories
(Multichannel, Urban) explicitly rather than an arbitrary integer threshold.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_FILE = (
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
RANDOM_STATE = 42

# Feature list used across all model stages.
# 12 features (v2). v1 had 9 features (label-encoded zip/channel).
FEATURES = [
    'recency',
    'log_history',
    'mens',
    'womens',
    'both_catalogs',
    'newbie',
    'history_segment_enc',          # ordinal: spend tiers have a genuine ordering
    'zip_Suburban',                 # 1 = Suburban customer, 0 = not (Rural is reference)
    'zip_Urban',                    # 1 = Urban customer, 0 = not
    'channel_Multichannel',         # 1 = Multichannel buyer, 0 = not (Phone is reference)
    'channel_Web',                  # 1 = Web buyer, 0 = not
    'recency_x_newbie',             # interaction: months_since_purchase * is_new_customer
]


def prepare_features(df):
    """
    Add all derived feature columns to a copy of df and return it.
    Call on the raw CSV load before slicing into treatment/control subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Raw load of DATA_FILE.

    Returns
    -------
    pd.DataFrame with all FEATURES columns present, plus original columns unchanged.
    """
    df = df.copy()

    # Log-transform history: right-skewed distribution, $29.99 floor
    df['log_history'] = np.log1p(df['history'])

    # Both-catalog overlap flag
    df['both_catalogs'] = (
        (df['mens'] == 1) & (df['womens'] == 1)
    ).astype(int)

    # Explicit recency x newbie interaction.
    # Captures: lapsed newbies respond differently to email than lapsed
    # established customers -- the most important interaction in the dataset
    # (see Stage 6b findings). The causal forest has both features marginally
    # but no multiplicative term to detect the interaction directly.
    df['recency_x_newbie'] = df['recency'] * df['newbie']

    # history_segment: ordinal label-encoding.
    # Alphabetical order of the tier labels approximates spend-tier order
    # well enough for tree-based models, and it has a genuine ordering
    # unlike zip_code or channel.
    le = LabelEncoder()
    df['history_segment_enc'] = le.fit_transform(df['history_segment'].astype(str))

    # zip_code: one-hot, Rural as reference category (dropped).
    # Avoids imposing a false ordinal structure (Urban > Suburban > Rural
    # or whatever alphabetical order assigns). Suburban and Urban respond
    # differently to Mens vs Womens email -- a split the ordinal encoding masks.
    zip_dummies = pd.get_dummies(df['zip_code'], prefix='zip')
    df['zip_Suburban'] = zip_dummies.get(
        'zip_Suburban', pd.Series(0, index=df.index)
    ).astype(int)
    df['zip_Urban'] = zip_dummies.get(
        'zip_Urban', pd.Series(0, index=df.index)
    ).astype(int)

    # channel: one-hot, Phone as reference category (weakest email responder).
    # Phone Mens lift: +0.55pp vs Multichannel: +1.02pp (see POSSIBLE_DIRECTIONS.md).
    # Phone customers show almost no Womens email response (+0.17pp).
    # One-hot lets the forest and policy tree treat Multichannel as a strong
    # moderator of treatment response rather than a marginal additive predictor.
    chan_dummies = pd.get_dummies(df['channel'], prefix='channel')
    df['channel_Multichannel'] = chan_dummies.get(
        'channel_Multichannel', pd.Series(0, index=df.index)
    ).astype(int)
    df['channel_Web'] = chan_dummies.get(
        'channel_Web', pd.Series(0, index=df.index)
    ).astype(int)

    return df
