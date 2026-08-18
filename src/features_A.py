"""
Block A — Static Venture Features
Preprocessing with:
  1. Missingness audit 
  2. BayesianRidge imputation
  3. Full spec alignment

Granularity: one row per Venture_ID x Cohort_Year.
Ventures admitted multiple times appear once per cohort - no deduplication.
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

    miss    = df.isnull().mean().rename("miss_rate")
    n_miss  = df.isnull().sum().rename("n_missing")
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
            print(f"  -> Dropping {len(dropped)} column(s): {dropped}")
        else:
            print("  -> No columns exceed the missingness threshold.")
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

def _binary_map(series: pd.Series, yes_val="Yes", no_val="No") -> pd.Series:
    return series.map({yes_val: 1, no_val: 0})


def _months_since_inc(cohort_df: pd.DataFrame) -> pd.Series:
    raw_inc = cohort_df.get("Incorporation_Date")
    if raw_inc is None:
        return pd.Series(np.nan, index=cohort_df.index)
    inc_date   = pd.to_datetime(raw_inc, errors="coerce")
    start_year = cohort_df["Cohort_Year"].astype(str).str.split("/").str[0]
    start_date = pd.to_datetime(start_year + "-10-01", errors="coerce")
    months     = ((start_date - inc_date) / np.timedelta64(1, "D")) / 30.4375
    months     = months.clip(upper=240)
    not_inc    = cohort_df.get(
        "Incorporated_PreCDL", pd.Series("Yes", index=cohort_df.index)
    )
    months = months.where(not_inc == "Yes", other=np.nan)
    return months


def _funding_velocity(total_funding: pd.Series, months: pd.Series) -> pd.Series:
    safe_months = months.clip(lower=1)
    return np.log1p(total_funding / safe_months)


# ─── 3 · Feature splitting ────────────────────────────────────────────────────

def split_by_year(
    app_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    cutoff_year: str = "2021/22",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_mask = cohort_df["Cohort_Year"] <= cutoff_year
    test_mask  = cohort_df["Cohort_Year"] > cutoff_year

    cohort_train = cohort_df[train_mask].reset_index(drop=True)
    cohort_test  = cohort_df[test_mask].reset_index(drop=True)

    train_ids = set(cohort_train["Venture_ID"])
    test_ids  = set(cohort_test["Venture_ID"])

    app_train = app_df[app_df["Venture_ID"].isin(train_ids)].reset_index(drop=True)
    app_test  = app_df[app_df["Venture_ID"].isin(test_ids)].reset_index(drop=True)

    print(f"Train (<= {cutoff_year}) : {len(cohort_train):,} rows")
    print(f"Test  (>  {cutoff_year}) : {len(cohort_test):,} rows")

    return app_train, app_test, cohort_train, cohort_test


# ─── 4 · Main builder ─────────────────────────────────────────────────────────

def build_block_A(
    app_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    miss_threshold: float = 0.50,
    imputer: IterativeImputer | None = None,
    fit_imputer: bool = True,
) -> tuple[pd.DataFrame, IterativeImputer, pd.DataFrame]:
    """
    Build Block A feature matrix.

    Granularity: one row per Venture_ID x Cohort_Year.
    Ventures admitted multiple times appear once per cohort.

    Parameters
    ----------
    app_df         : 01_Application_Data table
    cohort_df      : 03_Venture_Cohort table
    miss_threshold : raw columns with > this fraction missing are dropped
    imputer        : pre-fitted IterativeImputer; pass for test splits
    fit_imputer    : if True, fit imputer on this data (training split)

    Returns
    -------
    df       : engineered feature DataFrame (Venture_ID x Cohort_Year)
    imputer  : fitted IterativeImputer (reuse on test)
    audit_df : missingness audit table
    """

    # ── Base frame: one row per Venture_ID x Cohort_Year ─────────────────────
    df = cohort_df[
        ["Venture_ID", "Cohort_Year", "CDL_Site_Group", "CDL_Stream_Group"]
    ].copy().reset_index(drop=True)

    # ── A1 · Binary traction flags ────────────────────────────────────────────
    df["revenue_binary"]      = _binary_map(cohort_df["Revenue_Binary"].values,       )
    df["dilutive_binary"]     = _binary_map(cohort_df["Dilutive_Binary"].values,       )
    df["nondilutive_binary"]  = _binary_map(cohort_df["NonDilutive_Binary"].values,    )
    df["prototype_binary"]    = _binary_map(cohort_df["Prototype_Binary"].values,      )
    df["patent_binary"]       = _binary_map(cohort_df["Patent_Binary"].values,         )
    df["incorporated_precdl"] = _binary_map(cohort_df["Incorporated_PreCDL"].values,   )

    # ── A1 · Funding amounts ──────────────────────────────────────────────────
    dilutive_amt    = cohort_df["Dilutive_Amount_CAD"].copy().values
    nondilutive_amt = cohort_df["NonDilutive_Amount_CAD"].copy().values

    dilutive_amt    = np.where(df["dilutive_binary"]    != 0, dilutive_amt,    0.0)
    nondilutive_amt = np.where(df["nondilutive_binary"] != 0, nondilutive_amt, 0.0)

    df["log_dilutive_amt"]    = np.log1p(dilutive_amt)
    df["log_nondilutive_amt"] = np.log1p(nondilutive_amt)
    df["log_total_funding"]   = np.log1p(dilutive_amt + nondilutive_amt)

    # ── A1 · Employees ────────────────────────────────────────────────────────
    df["log_n_employees"] = np.log1p(cohort_df["N_Employees"].values)

    # ── A1 · Composite traction ───────────────────────────────────────────────
    df["has_any_traction"] = (
        (df["revenue_binary"] == 1) | (df["dilutive_binary"] == 1)
    ).astype(int)

    # ── A1 · Venture age & velocity ───────────────────────────────────────────
    df["months_since_inc"] = _months_since_inc(cohort_df).values
    df["funding_velocity"] = _funding_velocity(
        pd.Series(dilutive_amt + nondilutive_amt),
        df["months_since_inc"],
    ).values

    # ── A1 · Sparse / recent features ────────────────────────────────────────
    df["active_customers"] = _binary_map(
        cohort_df.get("Active_Customers", pd.Series(np.nan, index=cohort_df.index))
    )
    df["mlai_component"] = _binary_map(
        cohort_df.get("MLAI_Component_Binary", pd.Series(np.nan, index=cohort_df.index))
    )

    # ── A2 · Categorical pass-throughs ───────────────────────────────────────
    df["naics_supersector"]    = cohort_df["NAICS2022_GPT_SSect"].fillna("Unknown").values
    df["geo_grp"]              = cohort_df["Geo_Grp"].fillna("Unknown").values
    df["venture_stage_binary"] = cohort_df["Venture_Stage_Binary"].map(
        {"Early Stage": 1, "Late Stage": 0}
    ).values

    # ── A3 · Evaluation scores — merge on Venture_ID + Cohort_Year ───────────
    eval_cols = [
        "Venture_ID", "Cohort_Year",
        "Eval_AvgScore_Overall", "N_Evaluations",
        "N_RecInterview_Yes", "N_RecInterview_No",
        "N_Times_Applied",
    ]
    available_eval_cols = [c for c in eval_cols if c in app_df.columns]
    app_eval = app_df[available_eval_cols].copy()

    df = df.merge(app_eval, on=["Venture_ID", "Cohort_Year"], how="left")

    df["eval_avg_score"] = df.get("Eval_AvgScore_Overall", pd.Series(np.nan, index=df.index))
    df["eval_n_recs"]    = df.get("N_Evaluations",         pd.Series(np.nan, index=df.index))

    yes = df.get("N_RecInterview_Yes", pd.Series(0.0, index=df.index))
    no  = df.get("N_RecInterview_No",  pd.Series(0.0, index=df.index))
    df["eval_agreement_rate"] = np.where((yes + no) > 0, yes / (yes + no), np.nan)

    n_applied = df.get("N_Times_Applied", pd.Series(np.nan, index=df.index))
    df["is_repeat_applicant"] = (n_applied > 1).astype(float)
    df["n_prior_rejections"]  = (n_applied - 1).clip(lower=0)

    # Drop raw eval columns — keep only engineered ones
    raw_eval_drop = [
        c for c in ["Eval_AvgScore_Overall", "N_Evaluations",
                     "N_RecInterview_Yes", "N_RecInterview_No", "N_Times_Applied"]
        if c in df.columns
    ]
    df.drop(columns=raw_eval_drop, inplace=True)

    # ── A4 · Founder & team ───────────────────────────────────────────────────
    df["n_founders_cat"]        = cohort_df["N_Founders_Cat"].values
    df["female_founder_binary"] = cohort_df["Female_Founder_Binary"].values
    df["affiliations_binary"]   = _binary_map(cohort_df["Affiliations_Binary"].values)
    df["advisors_binary"]       = _binary_map(cohort_df["Advisors_Binary"].values)

    df["phd_ratio"] = (
        cohort_df["N_Founders_PhD"].values
        / pd.Series(cohort_df["N_Founders_Imputed"].values).replace(0, np.nan)
    ).values

    edu_map = {
        "Doctoral Degree": 6, "Professional Degree": 5,
        "Master's Degree": 4, "Bachelor's Degree": 3,
        "Certificate": 2,     "Unknown": 1,
    }
    df["highest_edu_ordinal"] = cohort_df["Highest_Venture_Edu"].map(edu_map).values

    # ── A5 · CDL history ──────────────────────────────────────────────────────
    df["n_times_admitted"] = cohort_df["N_Times_Admitted"].fillna(1).values

    help_cols = [c for c in cohort_df.columns if c.startswith("Help_")]
    if help_cols:
        help_bin   = cohort_df[help_cols].apply(lambda col: col.map({"Yes": 1, "No": 0}))
        df["n_help_areas"] = help_bin.sum(axis=1, min_count=1).values

    df = df.reset_index(drop=True)

    # ── Categorical imputation ────────────────────────────────────────────────
    if imputer is None:
        imputer = make_imputer()

    CAT_UNKNOWN = ["naics_supersector", "geo_grp", "CDL_Site_Group", "CDL_Stream_Group"]
    CAT_MODE    = ["n_founders_cat"]

    if fit_imputer:
        cat_modes = {
            col: df[col].mode()[0]
            for col in CAT_MODE
            if col in df.columns and df[col].isnull().any()
        }
        imputer._cat_modes = cat_modes
    else:
        cat_modes = imputer._cat_modes

    for col in CAT_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col, mode_val in cat_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)

    # ── Missingness audit on numeric features ─────────────────────────────────
    NUMERIC_COLS = [
        "revenue_binary", "dilutive_binary", "nondilutive_binary",
        "prototype_binary", "patent_binary", "incorporated_precdl",
        "venture_stage_binary", "active_customers", "mlai_component",
        "log_dilutive_amt", "log_nondilutive_amt", "log_total_funding",
        "log_n_employees", "months_since_inc", "funding_velocity",
        "has_any_traction",
        "eval_avg_score", "eval_n_recs", "eval_agreement_rate",
        "female_founder_binary", "affiliations_binary", "advisors_binary",
        "phd_ratio", "highest_edu_ordinal",
        "n_times_admitted", "is_repeat_applicant", "n_prior_rejections",
        "n_help_areas",
    ]
    impute_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if fit_imputer:
        audit_df, keep_cols, drop_cols = audit_missingness(
            df[impute_cols], threshold=miss_threshold, verbose=True
        )
        if drop_cols:
            print(f"\n  Removing high-missingness columns: {drop_cols}\n")
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
            impute_cols = [c for c in impute_cols if c not in drop_cols]
    else:
        impute_cols = imputer._block_a_impute_cols
        drop_cols   = [
            c for c in NUMERIC_COLS
            if c in df.columns and c not in impute_cols
        ]
        if drop_cols:
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        audit_df = pd.DataFrame()

    # ── BayesianRidge imputation ──────────────────────────────────────────────
    X_num = df[impute_cols].values

    if fit_imputer:
        X_imputed = imputer.fit_transform(X_num)
        imputer._block_a_impute_cols = impute_cols
    else:
        X_imputed = imputer.transform(X_num)

    df[impute_cols] = X_imputed

    BINARY_COLS = [
        c for c in impute_cols
        if any(kw in c for kw in ["binary", "flag", "missing", "traction", "repeat", "has_any"])
    ]
    df[BINARY_COLS] = df[BINARY_COLS].clip(0, 1).round()

    CAT_COLS = ["CDL_Site_Group", "CDL_Stream_Group", "naics_supersector", "geo_grp", "n_founders_cat"]
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df, imputer, audit_df


# ─── 5 · Quick smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n   = 200

    cohort_df = pd.DataFrame({
        "Venture_ID":             [f"V{i:03d}" for i in range(n)],
        "Cohort_Year":            rng.choice(["2019/20", "2020/21", "2021/22", "2022/23"], n),
        "CDL_Site_Group":         rng.choice(["Canada", "USA", "Other"], n),
        "CDL_Stream_Group":       rng.choice(["Health", "Tech", "Other"], n),
        "Revenue_Binary":         rng.choice(["Yes", "No", None], n, p=[.5, .4, .1]),
        "Dilutive_Binary":        rng.choice(["Yes", "No", None], n, p=[.4, .5, .1]),
        "NonDilutive_Binary":     rng.choice(["Yes", "No"], n),
        "Prototype_Binary":       rng.choice(["Yes", "No"], n),
        "Patent_Binary":          rng.choice(["Yes", "No"], n),
        "Incorporated_PreCDL":    rng.choice(["Yes", "No", None], n, p=[.7, .2, .1]),
        "Dilutive_Amount_CAD":    np.where(rng.random(n) < .5, rng.exponential(500_000, n), np.nan),
        "NonDilutive_Amount_CAD": np.where(rng.random(n) < .4, rng.exponential(200_000, n), np.nan),
        "N_Employees":            np.where(rng.random(n) < .85, rng.integers(1, 50, n).astype(float), np.nan),
        "NAICS2022_GPT_SSect":    rng.choice(["Tech", "Health", "Energy", None], n, p=[.4,.3,.2,.1]),
        "Geo_Grp":                rng.choice(["Canada", "US", "Europe", None], n, p=[.5,.3,.1,.1]),
        "Venture_Stage_Binary":   rng.choice(["Early Stage", "Late Stage", None], n, p=[.6,.3,.1]),
        "Female_Founder_Binary":  rng.choice([0, 1], n),
        "Affiliations_Binary":    rng.choice(["Yes", "No"], n),
        "Advisors_Binary":        rng.choice(["Yes", "No", None], n, p=[.5,.4,.1]),
        "N_Founders_Cat":         rng.choice(["1", "2", "3+"], n),
        "N_Founders_Imputed":     rng.integers(1, 4, n).astype(float),
        "N_Founders_PhD":         rng.integers(0, 3, n).astype(float),
        "Highest_Venture_Edu":    rng.choice(["Doctoral Degree", "Master's Degree", "Bachelor's Degree", "Unknown"], n),
        "N_Times_Admitted":       rng.integers(1, 3, n).astype(float),
        "Incorporation_Date":     pd.to_datetime([
            f"20{rng.integers(10,23):02d}-{rng.integers(1,13):02d}-01"
            if rng.random() > 0.1 else None
            for _ in range(n)
        ]),
        "Active_Customers": np.where(rng.random(n) > 0.7, rng.choice(["Yes", "No"], n), None),
    })

    app_df = pd.DataFrame({
        "Venture_ID":            [f"V{i:03d}" for i in range(n)],
        "Cohort_Year":           rng.choice(["2019/20", "2020/21", "2021/22", "2022/23"], n),
        "Eval_AvgScore_Overall": np.where(rng.random(n) < .9, rng.uniform(1, 5, n), np.nan),
        "N_Evaluations":         rng.integers(1, 6, n).astype(float),
        "N_RecInterview_Yes":    rng.integers(0, 5, n).astype(float),
        "N_RecInterview_No":     rng.integers(0, 3, n).astype(float),
        "N_Times_Applied":       rng.integers(1, 4, n).astype(float),
    })

    df_out, fitted_imputer, audit = build_block_A(
        app_df, cohort_df, miss_threshold=0.50
    )

    print(f"\nOutput shape : {df_out.shape}")
    print(f"Unique Venture_ID x Cohort_Year : {df_out[['Venture_ID','Cohort_Year']].drop_duplicates().shape[0]}")
    print(f"Remaining NaNs : {df_out.isnull().sum().sum()}")
    print(df_out.head())