"""
Block D — Historical Mentor Behavior Features
Preprocessing with:
  1. Missingness audit     
  2. BayesianRidge imputation 
  3. Full spec alignment:
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)


# ─── 0 · Missingness audit ────────────────────────────────────────────────────

def audit_missingness(df: pd.DataFrame, threshold: float = 0.50, verbose: bool = True) -> tuple[pd.DataFrame, list[str], list[str]]:
   
    miss = df.isnull().mean().rename("miss_rate")
    n_miss = df.isnull().sum().rename("n_missing")
    audit_df = pd.concat([n_miss, miss], axis=1).reset_index()
    audit_df.columns = ["column", "n_missing", "miss_rate"]
    audit_df["decision"] = np.where(audit_df["miss_rate"] > threshold, "DROP", "KEEP")
    audit_df = audit_df.sort_values("miss_rate", ascending=False).reset_index(drop=True)

    if verbose:
        print("=" * 65)
        print(f"MISSINGNESS AUDIT  (threshold = {threshold:.0%})")
        print("=" * 65)
        print(audit_df.to_string(index=False))
        print()
        dropped = audit_df[audit_df["decision"] == "DROP"]["column"].tolist()
        if dropped:
            print(f"  → Dropping {len(dropped)} column(s): {dropped}")
        else:
            print("  → No columns exceed the missingness threshold.")
        print("=" * 65)

    keep = audit_df[audit_df["decision"] == "KEEP"]["column"].tolist()
    drop = audit_df[audit_df["decision"] == "DROP"]["column"].tolist()
    return audit_df, keep, drop


# ─── 1 · BayesianRidge imputer ────────────────────────────────────────────────

def make_imputer(max_iter: int = 10, random_state: int = 42, initial_strategy: str = "median") -> IterativeImputer:
   
    return IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy=initial_strategy,
        imputation_order="ascending",   
    )


def impute_numeric(
    df: pd.DataFrame,
    num_cols: list[str],
    imputer: Optional[IterativeImputer] = None,
    fit_imputer: bool = True,
) -> tuple[pd.DataFrame, IterativeImputer]:
    if not num_cols:
        return df.copy(), imputer or make_imputer()
    df = df.copy()
    if fit_imputer:
        imputer = make_imputer()
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        if imputer is None:
            raise ValueError("fit_imputer=False mais aucun imputer fourni.")
        df[num_cols] = imputer.transform(df[num_cols])
    return df, imputer


def _clean_sentinels(series: pd.Series, sentinel: float = -2.0) -> pd.Series:
    return series.replace(sentinel, np.nan)


# ─── 2 · D1 — Mentor Selectivity ─────────────────────────────────────────────

def build_D1_selectivity(
    hands_df: pd.DataFrame,
    sgm_reg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    avg_mentor_hist_handrate : mean historical hand-raise rate across SGM mentors.
    min_mentor_hist_handrate : min historical hand-raise rate (most selective mentor).

    Key for joining : Venture_ID, Session_Num, Cohort_Year (present in both tables).
    LEAKAGE: sessions strictly earlier (Cohort_Year < current OR same year + Session_Num < current).
    """
    hands = hands_df[hands_df["Commitment_Type"] == "Mentorship"].copy()

    attended = (
        sgm_reg_df
        .groupby(["Person_ID", "Cohort_Year", "Session_Num"])
        .agg(ventures_attended=("Venture_ID", "nunique"))
        .reset_index()
    )
    raised = (
        hands  # déjà filtré Commitment_Type == "Mentorship"
        .groupby(["Person_ID", "Cohort_Year", "Session_Num"])
        .agg(hands_raised=("Venture_ID", "nunique"))
        .reset_index()
    )
    mentor_hist = attended.merge(raised, on=["Person_ID", "Cohort_Year", "Session_Num"], how="left")
    mentor_hist["hands_raised"] = mentor_hist["hands_raised"].fillna(0)

    results = []
    vs_keys = sgm_reg_df[["Venture_ID", "Session_Num", "Cohort_Year"]].drop_duplicates()

    for _, row in vs_keys.iterrows():
        vid, snum, cyear = row["Venture_ID"], row["Session_Num"], row["Cohort_Year"]

        mentors = sgm_reg_df[
            (sgm_reg_df["Venture_ID"] == vid)
            & (sgm_reg_df["Session_Num"] == snum)
            & (sgm_reg_df["Cohort_Year"] == cyear)
        ]["Person_ID"].unique()

        if len(mentors) == 0:
            results.append({"Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
                             "avg_mentor_hist_handrate": np.nan, "min_mentor_hist_handrate": np.nan})
            continue

        prior = mentor_hist[
            (mentor_hist["Person_ID"].isin(mentors))
            & (
                (mentor_hist["Cohort_Year"] < cyear)
                | ((mentor_hist["Cohort_Year"] == cyear) & (mentor_hist["Session_Num"] < snum))
            )
        ]

        if prior.empty:
            results.append({"Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
                             "avg_mentor_hist_handrate": np.nan, "min_mentor_hist_handrate": np.nan})
            continue

        mentor_rates = (
            prior.groupby("Person_ID")
            .agg(total_attended=("ventures_attended", "sum"),
                 total_hands=("hands_raised", "sum"))
            .assign(hand_rate=lambda x: x["total_hands"] / x["total_attended"].clip(lower=1))
        )

        results.append({
            "Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
            "avg_mentor_hist_handrate": mentor_rates["hand_rate"].mean(),
            "min_mentor_hist_handrate": mentor_rates["hand_rate"].min(),
        })

    return pd.DataFrame(results)


def build_D1_experience(
    mentor_cohort_df: pd.DataFrame,
    sgm_reg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    avg_mentor_experience   : mean N_ActiveYrs_CDL among SGM mentors.
    pct_experienced_mentors : share with N_ActiveYrs_CDL > 3.

    Key for joining : Person_ID + Cohort_Year (present in mentor_cohort and sgm_reg).
    """
    mentor_exp = mentor_cohort_df[["Person_ID", "Cohort_Year", "N_ActiveYrs_CDL"]].copy()
    mentor_exp["N_ActiveYrs_CDL"] = _clean_sentinels(mentor_exp["N_ActiveYrs_CDL"])

    merged = sgm_reg_df.merge(mentor_exp, on=["Person_ID", "Cohort_Year"], how="left")

    d1_exp = (
        merged
        .groupby(["Venture_ID", "Session_Num", "Cohort_Year"])
        .agg(
            avg_mentor_experience=("N_ActiveYrs_CDL", "mean"),
            pct_experienced_mentors=(
                "N_ActiveYrs_CDL",
                lambda x: (x > 3).sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan
            ),
        )
        .reset_index()
    )
    return d1_exp


# ─── 3 · D2 — Domain Persistence ─────────────────────────────────────────────

def build_D2_domain_persistence(
    hands_df: pd.DataFrame,
    sgm_reg_df: pd.DataFrame,
    venture_session_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    avg_domain_persistence : for each SGM mentor, share of historical hands raised for ventures in the same Stream_ID. Average across SGM mentors.

    Stream_ID is available in 09_Hands_Raised and 05_Venture_Session.
    LEAKAGE: sessions strictly earlier.
    """
    hands = hands_df[hands_df["Commitment_Type"] == "Mentorship"].copy()

    # Stream_ID de la venture courante depuis venture_session_df
    stream_map = (
        venture_session_df[["Venture_ID", "Cohort_Year", "Stream_ID"]]
        .dropna(subset=["Stream_ID"])
        .drop_duplicates(subset=["Venture_ID", "Cohort_Year"])
    )

    results = []
    vs_keys = (
        sgm_reg_df[["Venture_ID", "Session_Num", "Cohort_Year"]]
        .drop_duplicates()
        .merge(stream_map, on=["Venture_ID", "Cohort_Year"], how="left")
    )

    for _, row in vs_keys.iterrows():
        vid, snum, cyear = row["Venture_ID"], row["Session_Num"], row["Cohort_Year"]
        current_stream = row.get("Stream_ID", None)

        mentors = sgm_reg_df[
            (sgm_reg_df["Venture_ID"] == vid)
            & (sgm_reg_df["Session_Num"] == snum)
            & (sgm_reg_df["Cohort_Year"] == cyear)
        ]["Person_ID"].unique()

        if len(mentors) == 0 or pd.isna(current_stream):
            results.append({"Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
                             "avg_domain_persistence": np.nan})
            continue

        prior_hands = hands[
            (hands["Person_ID"].isin(mentors))
            & (
                (hands["Cohort_Year"] < cyear)
                | ((hands["Cohort_Year"] == cyear) & (hands["Session_Num"] < snum))
            )
        ]

        if prior_hands.empty:
            results.append({"Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
                             "avg_domain_persistence": np.nan})
            continue

        mentor_persistence = (
            prior_hands
            .groupby("Person_ID")
            .apply(lambda g: (g["Stream_ID"] == current_stream).mean())
        )

        results.append({
            "Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
            "avg_domain_persistence": mentor_persistence.mean(),
        })

    return pd.DataFrame(results)


# ─── 4 · D3 — Within-Session Fatigue ─────────────────────────────────────────

def build_D3_fatigue(
    hands_df: pd.DataFrame,
    sgm_reg_df: pd.DataFrame,
    venture_session_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    mentor_hands_so_far : raised hands by SGM mentors for the ventures with a lower LRD_Order in the same session. Average across mentors.

    Key: Venture_ID + Session_Num + Cohort_Year.
    """
    hands = hands_df[hands_df["Commitment_Type"] == "Mentorship"].copy()
    hands["raised"] = 1  

    lrd_order = venture_session_df[
        ["Venture_ID", "Session_Num", "Cohort_Year", "LRD_Order"]
    ].dropna(subset=["LRD_Order"])

    # Attach LRD_Order to hands via Venture_ID + Session_Num + Cohort_Year
    hands = hands.merge(lrd_order, on=["Venture_ID", "Session_Num", "Cohort_Year"], how="left")

    results = []
    vs_keys = venture_session_df[
        ["Venture_ID", "Session_Num", "Cohort_Year"]
    ].drop_duplicates().merge(lrd_order, on=["Venture_ID", "Session_Num", "Cohort_Year"], how="left")

    for _, row in vs_keys.iterrows():
        vid, snum, cyear, order = row["Venture_ID"], row["Session_Num"], row["Cohort_Year"], row["LRD_Order"]

        mentors = sgm_reg_df[
            (sgm_reg_df["Venture_ID"] == vid)
            & (sgm_reg_df["Session_Num"] == snum)
            & (sgm_reg_df["Cohort_Year"] == cyear)
        ]["Person_ID"].unique()

        if len(mentors) == 0 or pd.isna(order):
            results.append({"Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
                             "mentor_hands_so_far": np.nan})
            continue

        prior_in_session = hands[
            (hands["Person_ID"].isin(mentors))
            & (hands["Session_Num"] == snum)
            & (hands["Cohort_Year"] == cyear)
            & (hands["LRD_Order"] < order)
        ]

        if prior_in_session.empty:
            results.append({"Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
                             "mentor_hands_so_far": 0.0})
            continue

        per_mentor = prior_in_session.groupby("Person_ID")["raised"].sum()
        results.append({
            "Venture_ID": vid, "Session_Num": snum, "Cohort_Year": cyear,
            "mentor_hands_so_far": per_mentor.reindex(mentors).mean(),
        })

    return pd.DataFrame(results)


# ─── 5 · D4 — S1 Lagged Signal (S2 rows only) ────────────────────────────────

def build_D4_lagged_s1(
    hands_venture_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    hands_received_s1 : mentorship hands received at Session 1.
    s1_hand_rate      : hands_received_s1 / n_mentors_present_S1.
    NaN for Session_Num == 1.
    """
    hands = hands_venture_df[hands_venture_df["Commitment_Type"] == "Mentorship"].copy()

    mentor_type_filter = (
        hands["Mentor_Type_Cat"] != 0
        if "Mentor_Type_Cat" in hands.columns
        else pd.Series(True, index=hands.index)
    )

    s1_hands = (
        hands[
            (hands["Session_Num"] == 1)
            & mentor_type_filter
        ]
        .groupby(["Venture_ID", "Cohort_Year"])
        .agg(hands_received_s1=("Person_ID", "nunique"))
        .reset_index()
    )

    n_mentors_s1 = (
        hands[hands["Session_Num"] == 1]
        .groupby(["Venture_ID", "Cohort_Year"])
        .agg(n_mentors_s1=("Person_ID", "nunique"))
        .reset_index()
    )

    d4 = s1_hands.merge(n_mentors_s1, on=["Venture_ID", "Cohort_Year"], how="left")
    d4["s1_hand_rate"] = d4["hands_received_s1"] / d4["n_mentors_s1"].clip(lower=1)
    d4["Session_Num"] = 2

    return d4[["Venture_ID", "Cohort_Year", "Session_Num", "hands_received_s1", "s1_hand_rate"]]


# ─── 6 · Main entry point ─────────────────────────────────────────────────────

def build_block_D(
    hands_df: pd.DataFrame,
    hands_venture_df: pd.DataFrame,
    sgm_reg_df: pd.DataFrame,
    mentor_cohort_df: pd.DataFrame,
    venture_session_df: pd.DataFrame,
    imputer: Optional[IterativeImputer] = None,
    fit_imputer: bool = True,
    missingness_threshold: float = 0.50,
    verbose: bool = True,
) -> tuple[pd.DataFrame, IterativeImputer, pd.DataFrame]:
    """
    Builds the complete feature matrix for Block D.

    Parameters
    ----------
    hands_df           : 09_Hands_Raised
    sgm_reg_df         : 08_SGM_Registrations  (doit avoir Cohort_Year)
    mentor_cohort_df   : 04_Mentor_Cohort
    venture_session_df : 05_Venture_Session
    imputer            : IterativeImputer pré-fitté (passer pour le test set).
    fit_imputer        : True → fit sur ces données ; False → transform seulement.
    """

    max_cohort_year = venture_session_df["Cohort_Year"].max()
    mentor_cohort_df = mentor_cohort_df[
        mentor_cohort_df["Cohort_Year"] <= max_cohort_year
    ].reset_index(drop=True)

    # SGM registrations n'a pas Cohort_Year — on le récupère depuis hands_df
    if "Cohort_Year" not in sgm_reg_df.columns:
        cohort_year_map = (
            hands_df[["Cohort_ID", "Cohort_Year"]]
            .dropna()
            .drop_duplicates()
        )
        sgm_reg_df = sgm_reg_df.merge(cohort_year_map, on="Cohort_ID", how="left")

    KEY = ["Venture_ID", "Session_Num", "Cohort_Year"]

    # ── D1 ────────────────────────────────────────────────────────────────────
    d1_sel = build_D1_selectivity(hands_df, sgm_reg_df)
    d1_exp = build_D1_experience(mentor_cohort_df, sgm_reg_df)
    d1 = d1_sel.merge(d1_exp, on=KEY, how="outer")

    # ── D2 ────────────────────────────────────────────────────────────────────
    d2 = build_D2_domain_persistence(hands_df, sgm_reg_df, venture_session_df)

    # ── D3 ────────────────────────────────────────────────────────────────────
    d3 = build_D3_fatigue(hands_df, sgm_reg_df, venture_session_df)

    # ── D4 ────────────────────────────────────────────────────────────────────
    d4 = build_D4_lagged_s1(hands_df)

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = d1.merge(d2, on=KEY, how="outer").merge(d3, on=KEY, how="outer")
    df = df.merge(d4, on=KEY, how="left")

    d4_cols = ["hands_received_s1", "s1_hand_rate"]

    if "hands_received_s1" in df.columns:
        df["hands_received_s1"] = df["hands_received_s1"].fillna(0)
    if "s1_hand_rate" in df.columns:
        df["s1_hand_rate"] = df["s1_hand_rate"].fillna(0)

    # ── Missingness audit + imputation ───────────────────────────────────────
    NUMERIC_COLS = [
        "avg_mentor_hist_handrate", "min_mentor_hist_handrate",
        "avg_mentor_experience", "pct_experienced_mentors",
        "avg_domain_persistence",
        "mentor_hands_so_far",
        "hands_received_s1", "s1_hand_rate",
    ]
    impute_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if fit_imputer:
        audit_df, keep_cols, drop_cols = audit_missingness(df[impute_cols], threshold=missingness_threshold, verbose=verbose)
        drop_cols = [c for c in drop_cols if c not in d4_cols]
        if drop_cols:
            print(f"\n  Removing high-missingness columns: {drop_cols}\n")
            df.drop(columns=drop_cols, inplace=True)
            impute_cols = [c for c in impute_cols if c not in drop_cols]
        imputer = make_imputer()
        df[impute_cols] = imputer.fit_transform(df[impute_cols])
        imputer._block_d_impute_cols = impute_cols
    else:
        audit_df = pd.DataFrame()
        impute_cols = imputer._block_d_impute_cols
        drop_cols = [c for c in NUMERIC_COLS if c in df.columns and c not in impute_cols]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        df[impute_cols] = imputer.transform(df[impute_cols])

    # ── Clip rates ────────────────────────────────────────────────────────────
    rate_cols = ["avg_mentor_hist_handrate", "min_mentor_hist_handrate",
                 "pct_experienced_mentors", "avg_domain_persistence", "s1_hand_rate"]
    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    return df, imputer, audit_df