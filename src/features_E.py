"""
Block E — Network Features
Preprocessing with:
  1. Missingness audit
  2. BayesianRidge imputation
  3. Full spec alignment
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# ─── 0 · Missingness audit ────────────────────────────────────────────────────

def audit_missingness(
    df: pd.DataFrame,
    threshold: float = 0.50,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:

    miss   = df.isnull().mean().rename("miss_rate")
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

def make_imputer(
    max_iter: int = 10,
    random_state: int = 42,
    initial_strategy: str = "median",
) -> IterativeImputer:

    return IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy=initial_strategy,
        imputation_order="ascending",
    )


# ─── 2 · Feature construction helpers ────────────────────────────────────────

def _build_cohand_lookup(
    hands_df: pd.DataFrame,
    current_cohort_year: str,
) -> pd.DataFrame | None:
    """
    Vectorised co-hand rate matrix.

    For every (mentor_A, mentor_B) pair, computes the fraction of prior venture-sessions in which both raised a Mentorship hand.
    Filters to Cohort_Year < current_cohort_year — no leakage.

    Returns a square DataFrame (mentors x mentors) of co-hand rates, or None if no prior data exists.
    """
    prior = hands_df[
        (hands_df["Commitment_Type"] == "Mentorship") &
        (hands_df["Cohort_Year"] < current_cohort_year)
    ][["Venture_ID", "Session_ID", "Person_ID"]].drop_duplicates()

    if prior.empty:
        return None

    pivot = (
        prior
        .assign(raised=1)
        .pivot_table(
            index=["Venture_ID", "Session_ID"],
            columns="Person_ID",
            values="raised",
            aggfunc="max",
            fill_value=0,
        )
    )

    n_sessions = len(pivot)
    if n_sessions == 0:
        return None

    # vectorised co-occurrence via matrix multiplication
    M = pivot.values.astype(np.float32)
    cooccur = M.T @ M
    cohand_matrix = cooccur / n_sessions

    return pd.DataFrame(
        cohand_matrix,
        index=pivot.columns,
        columns=pivot.columns,
    )


def _clique_cohand_rate(sgm_mentors: list, cohand_matrix: pd.DataFrame | None) -> float:
    """
    Mean pairwise historical co-hand rate for this SGM's mentor group.
    Returns NaN when fewer than 2 mentors are present or no history exists.
    """
    if cohand_matrix is None or len(sgm_mentors) < 2:
        return np.nan

    known = [m for m in sgm_mentors if m in cohand_matrix.index]
    if len(known) < 2:
        return np.nan

    sub = cohand_matrix.loc[known, known].values
    n   = len(known)
    idx = np.triu_indices(n, k=1)
    rates = sub[idx]

    return float(rates.mean()) if len(rates) > 0 else np.nan


# ─── 3 · E2 helpers (vectorised) ─────────────────────────────────────────────

def _build_e2_features(
    venture_session_df: pd.DataFrame,
    sgm_df: pd.DataFrame,
    hands_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vectorised computation of E2 features for all S2 ventures at once.
    Returns a DataFrame with Venture_ID, Cohort_Year and E2 columns.
    S1 ventures are not included — they will be filled with 0 afterwards.
    """
    sgm = sgm_df[["Venture_ID", "Cohort_Year", "Session_Num", "Person_ID"]].drop_duplicates()

    s1_sgm = sgm[sgm["Session_Num"] == 1].rename(columns={"Person_ID": "Person_ID_s1"})
    s2_sgm = sgm[sgm["Session_Num"] == 2].rename(columns={"Person_ID": "Person_ID_s2"})

    s2_ventures = venture_session_df[
        venture_session_df["Session_Num"] == 2
    ][["Venture_ID", "Cohort_Year"]]

    if s2_ventures.empty:
        return s2_ventures.assign(
            n_returning_mentors_s2=0,
            n_new_mentors_s2=0,
            returning_hand_s1_count=0,
        )

    s2 = s2_ventures.merge(
        s2_sgm[["Venture_ID", "Cohort_Year", "Person_ID_s2"]],
        on=["Venture_ID", "Cohort_Year"], how="left",
    )
    s2 = s2.merge(
        s1_sgm[["Venture_ID", "Cohort_Year", "Person_ID_s1"]],
        on=["Venture_ID", "Cohort_Year"], how="left",
    )

    s2["is_returning"] = s2["Person_ID_s2"] == s2["Person_ID_s1"]

    s1_hands = hands_df[
        (hands_df["Session_Num"] == 1) &
        (hands_df["Commitment_Type"] == "Mentorship")
    ][["Venture_ID", "Cohort_Year", "Person_ID"]].drop_duplicates()
    s1_hands = s1_hands.rename(columns={"Person_ID": "Person_ID_hand"})

    s2 = s2.merge(
        s1_hands,
        left_on=["Venture_ID", "Cohort_Year", "Person_ID_s2"],
        right_on=["Venture_ID", "Cohort_Year", "Person_ID_hand"],
        how="left",
    )
    s2["raised_s1_hand"] = s2["Person_ID_hand"].notna() & s2["is_returning"]

    agg = (
        s2.groupby(["Venture_ID", "Cohort_Year"])
        .agg(
            n_s2_mentors     =("Person_ID_s2",  "nunique"),
            n_returning      =("is_returning",   "sum"),
            n_returning_hand =("raised_s1_hand", "sum"),
        )
        .reset_index()
    )
    agg["n_returning_mentors_s2"]  = agg["n_returning"]
    agg["n_new_mentors_s2"]        = agg["n_s2_mentors"] - agg["n_returning"]
    agg["returning_hand_s1_count"] = agg["n_returning_hand"]

    return agg[["Venture_ID", "Cohort_Year",
                "n_returning_mentors_s2", "n_new_mentors_s2", "returning_hand_s1_count"]]


# ─── 4 · Train / test split helper ───────────────────────────────────────────

def split_by_year(
    venture_session_df: pd.DataFrame,
    cutoff_year: str = "2021/22",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split venture_session_df into train (<= cutoff) and test (> cutoff)."""
    train = venture_session_df[
        venture_session_df["Cohort_Year"] <= cutoff_year
    ].reset_index(drop=True)
    test = venture_session_df[
        venture_session_df["Cohort_Year"] > cutoff_year
    ].reset_index(drop=True)
    print(f"Train (<= {cutoff_year}): {len(train):,} venture-sessions")
    print(f"Test  (>  {cutoff_year}): {len(test):,} venture-sessions")
    return train, test


# ─── 5 · Main builder ─────────────────────────────────────────────────────────

_S2_ONLY_COLS = ["n_returning_mentors_s2", "n_new_mentors_s2", "returning_hand_s1_count"]
_GLOBAL_COLS  = ["clique_cohand_rate"]
NUMERIC_COLS  = _GLOBAL_COLS + _S2_ONLY_COLS


def build_block_E(
    venture_session_df: pd.DataFrame,
    sgm_df: pd.DataFrame,
    hands_df: pd.DataFrame,           # entière — pour E1 (historique mentors)
    hands_venture_df: pd.DataFrame,   # filtrée par Venture_ID — pour E2
    miss_threshold: float = 0.50,
    imputer: IterativeImputer | None = None,
    fit_imputer: bool = True,
) -> tuple[pd.DataFrame, IterativeImputer, pd.DataFrame]:
    """
    Build Block E feature matrix.

    Parameters
    ----------
    venture_session_df : 05_Venture_Session rows of interest.
                         Must contain: Venture_ID, Cohort_Year, Session_Num.
    sgm_df             : 08_SGM_Registrations (full table).
                         Must contain: Venture_ID, Cohort_Year, Session_Num, Person_ID.
    hands_df           : 09_Hands_Raised (full table).
                         Must contain: Venture_ID, Cohort_Year, Session_Num,
                         Session_ID, Person_ID, Commitment_Type.
    miss_threshold     : columns with > this fraction missing are dropped (default 50 %).
    imputer            : pre-fitted IterativeImputer; pass for val/test splits.
    fit_imputer        : if True, fit imputer on this data (training split).

    Returns
    -------
    df        : engineered feature DataFrame (one row per Venture_ID x Session_Num)
    imputer   : fitted IterativeImputer (reuse on val/test)
    audit_df  : missingness audit table (empty DataFrame on val/test)

    Notes
    -----
    E2 columns are 0 on S1 rows by convention - a venture that has not yet
    reached S2 has 0 returning mentors by definition, and this encoding allows
    a single unified audit and imputation pass across both sessions.
    """

    # ── Normalise ID types to avoid silent join mismatches ────────────────────
    for df_ in [venture_session_df, sgm_df, hands_df, hands_venture_df]:
        for col in ["Venture_ID", "Person_ID"]:
            if col in df_.columns:
                df_[col] = df_[col].astype(str)

    # ── E1 · clique_cohand_rate ───────────────────────────────────────────────
    cohort_years   = venture_session_df["Cohort_Year"].unique()
    cohand_by_year = {cy: _build_cohand_lookup(hands_df, cy) for cy in cohort_years}

    sgm_mentors = (
        sgm_df[["Venture_ID", "Cohort_Year", "Session_Num", "Person_ID"]]
        .drop_duplicates()
        .groupby(["Venture_ID", "Cohort_Year", "Session_Num"])["Person_ID"]
        .apply(list)
        .reset_index()
        .rename(columns={"Person_ID": "mentor_list"})
    )

    df = venture_session_df[["Venture_ID", "Cohort_Year", "Session_Num"]].copy()
    df = df.merge(sgm_mentors, on=["Venture_ID", "Cohort_Year", "Session_Num"], how="left")

    df["clique_cohand_rate"] = df.apply(
        lambda r: _clique_cohand_rate(
            r["mentor_list"] if isinstance(r["mentor_list"], list) else [],
            cohand_by_year.get(r["Cohort_Year"]),
        ),
        axis=1,
    )
    df.drop(columns=["mentor_list"], inplace=True)

    # ── E2 · Returning mentor features ───────────────────────────────────────
    e2 = _build_e2_features(venture_session_df, sgm_df, hands_venture_df)
    df = df.merge(e2, on=["Venture_ID", "Cohort_Year"], how="left")

    s1_mask = df["Session_Num"] == 1
    df.loc[s1_mask, _S2_ONLY_COLS] = df.loc[s1_mask, _S2_ONLY_COLS].fillna(0)

    df = df.reset_index(drop=True)

    # ── Initialise imputer ────────────────────────────────────────────────────
    if imputer is None:
        imputer = make_imputer()

    # ── Missingness audit — single pass on all rows and all columns ───────────
    if fit_imputer:
        numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
        audit_df, _, drop_cols = audit_missingness(
            df[numeric_present], threshold=miss_threshold, verbose=True,
        )

        if drop_cols:
            print(f"\n  Removing high-missingness columns: {drop_cols}\n")
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        imputer._block_e_impute_cols = [
            c for c in numeric_present if c not in drop_cols and c in df.columns
        ]

    else:
        # val/test: use exact columns seen at training time
        impute_cols = imputer._block_e_impute_cols
        extra = [c for c in NUMERIC_COLS if c in df.columns and c not in impute_cols]
        if extra:
            df.drop(columns=extra, inplace=True)
        audit_df = pd.DataFrame()

    impute_cols = imputer._block_e_impute_cols

    # ── BayesianRidge imputation — single pass on all rows ───────────────────
    if impute_cols:
        X = df[impute_cols].values
        if fit_imputer:
            df[impute_cols] = imputer.fit_transform(X)
        else:
            df[impute_cols] = imputer.transform(X)

        # clip count features to non-negative integers
        count_cols = [c for c in impute_cols if c in _S2_ONLY_COLS]
        if count_cols:
            df[count_cols] = df[count_cols].clip(lower=0).round().astype(float)

    return df, imputer, audit_df


# ─── 6 · Quick smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    rng          = np.random.default_rng(0)
    n_ventures   = 60
    mentor_pool  = [str(i) for i in range(30)]
    cohort_years = ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]

    vs_rows = []
    for v in range(n_ventures):
        cy = rng.choice(cohort_years)
        for snum in [1, 2]:
            vs_rows.append({"Venture_ID": str(v), "Cohort_Year": cy, "Session_Num": snum})
    venture_session_df = pd.DataFrame(vs_rows)

    sgm_rows = []
    for _, r in venture_session_df.iterrows():
        n_m = rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.20, 0.35, 0.25, 0.10])
        for m in rng.choice(mentor_pool, n_m, replace=False):
            sgm_rows.append({
                "Venture_ID": r["Venture_ID"], "Cohort_Year": r["Cohort_Year"],
                "Session_Num": r["Session_Num"], "Person_ID": m,
            })
    sgm_df = pd.DataFrame(sgm_rows)

    historical = ["2016/17", "2017/18"]
    hands_rows = []
    for cy in historical:
        for v in range(30):
            for snum in [1, 2]:
                for m in rng.choice(mentor_pool, rng.integers(0, 6), replace=False):
                    hands_rows.append({
                        "Venture_ID": str(v), "Cohort_Year": cy, "Session_Num": snum,
                        "Session_ID": f"{cy}_{snum}", "Person_ID": m,
                        "Commitment_Type": "Mentorship",
                    })
    for _, r in venture_session_df[venture_session_df["Session_Num"] == 1].iterrows():
        for m in rng.choice(mentor_pool, rng.integers(0, 4), replace=False):
            hands_rows.append({
                "Venture_ID": r["Venture_ID"], "Cohort_Year": r["Cohort_Year"],
                "Session_Num": 1, "Session_ID": f"{r['Cohort_Year']}_1",
                "Person_ID": m, "Commitment_Type": "Mentorship",
            })
    hands_df = pd.DataFrame(hands_rows)

    vs_train, vs_test = split_by_year(venture_session_df, cutoff_year="2021/22")

    print("\n=== TRAINING SPLIT ===")
    df_train, fitted_imputer, audit = build_block_E(
        vs_train, sgm_df, hands_df, miss_threshold=0.50
    )
    print(f"\nOutput shape: {df_train.shape}")
    print(df_train.head(8).to_string(index=False))
    print("\nRemaining NaNs (train):")
    print(df_train.isnull().sum())

    print("\n=== TEST SPLIT ===")
    df_test, _, _ = build_block_E(
        vs_test, sgm_df, hands_df,
        imputer=fitted_imputer, fit_imputer=False,
    )
    print(f"\nOutput shape: {df_test.shape}")
    print("\nRemaining NaNs (test):")
    print(df_test.isnull().sum())