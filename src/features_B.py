"""
Block B — SGM Engagement Features
Preprocessing with:
  1. Missingness audit
  2. BayesianRidge imputation
  3. Full spec alignment
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


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


# ─── 1 · BayesianRidge imputer ───────────────────────────────────────────────

def make_imputer(max_iter: int = 10, random_state: int = 42, initial_strategy: str = "median") -> IterativeImputer:

    return IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy=initial_strategy,
        imputation_order="ascending",
    )


# ─── 2 · Domain-match taxonomy ───────────────────────────────────────────────

_INDUSTRY_TO_SUPERSECTOR: dict[str, str] = {
    # Goods Producing
    "manufacturing":                    "1_Goods Producing Industries",
    "agriculture":                      "1_Goods Producing Industries",
    "mining":                           "1_Goods Producing Industries",
    "construction":                     "1_Goods Producing Industries",
    "aerospace":                        "1_Goods Producing Industries",
    "aerospace and defense":            "1_Goods Producing Industries",
    "defence":                          "1_Goods Producing Industries",
    "defense":                          "1_Goods Producing Industries",
    "hardware":                         "1_Goods Producing Industries",
    "space technology":                 "1_Goods Producing Industries",
    "deep tech":                        "1_Goods Producing Industries",
    # Trade / Transport / Utilities
    "retail":                           "2_Trade, Transportation, and Utilities",
    "wholesale":                        "2_Trade, Transportation, and Utilities",
    "transportation":                   "2_Trade, Transportation, and Utilities",
    "logistics":                        "2_Trade, Transportation, and Utilities",
    "utilities":                        "2_Trade, Transportation, and Utilities",
    "energy":                           "2_Trade, Transportation, and Utilities",
    "energy and sustainability":        "2_Trade, Transportation, and Utilities",
    "cleantech":                        "2_Trade, Transportation, and Utilities",
    "food and beverage":                "2_Trade, Transportation, and Utilities",
    # Information
    "information technology":           "3_Information",
    "technology":                       "3_Information",
    "tech":                             "3_Information",
    "software":                         "3_Information",
    "saas":                             "3_Information",
    "ai":                               "3_Information",
    "media":                            "3_Information",
    "telecommunications":               "3_Information",
    "telecom":                          "3_Information",
    "internet":                         "3_Information",
    "cybersecurity":                    "3_Information",
    "data":                             "3_Information",
    # Financial
    "finance":                          "4_Financial Activities",
    "financial services":               "4_Financial Activities",
    "banking":                          "4_Financial Activities",
    "insurance":                        "4_Financial Activities",
    "venture capital":                  "4_Financial Activities",
    "venture capital / technology":     "4_Financial Activities",
    "venture capital / deep tech":      "4_Financial Activities",
    "venture capital / finance":        "4_Financial Activities",
    "venture capital / software":       "4_Financial Activities",
    "private equity":                   "4_Financial Activities",
    "investment":                       "4_Financial Activities",
    "fintech":                          "4_Financial Activities",
    "real estate":                      "4_Financial Activities",
    # Professional & Business Services
    "consulting":                       "5_Professional and Business Services",
    "professional services":            "5_Professional and Business Services",
    "legal":                            "5_Professional and Business Services",
    "accounting":                       "5_Professional and Business Services",
    "marketing":                        "5_Professional and Business Services",
    "management":                       "5_Professional and Business Services",
    # Education & Health
    "health care":                      "6_Education and Health Services",
    "healthcare":                       "6_Education and Health Services",
    "biotech":                          "6_Education and Health Services",
    "biotech/healthcare":               "6_Education and Health Services",
    "biotechnology":                    "6_Education and Health Services",
    "life sciences":                    "6_Education and Health Services",
    "pharmaceuticals":                  "6_Education and Health Services",
    "medical":                          "6_Education and Health Services",
    "education":                        "6_Education and Health Services",
    "edtech":                           "6_Education and Health Services",
    "ai, healthcare":                   "6_Education and Health Services",
    # Other
    "government":                       "7_Other Services and Government",
    "non-profit":                       "7_Other Services and Government",
    "nonprofit":                        "7_Other Services and Government",
    "entertainment":                    "7_Other Services and Government",
    "hospitality":                      "7_Other Services and Government",
    "sports":                           "7_Other Services and Government",
}


def _mentor_industry_to_supersector(industry: str | None) -> str | None:
    if pd.isna(industry) or not isinstance(industry, str):
        return None
    key = industry.strip().lower()
    # Exact match first
    if key in _INDUSTRY_TO_SUPERSECTOR:
        return _INDUSTRY_TO_SUPERSECTOR[key]
    # Substring match (longest-key-first to prefer specificity)
    for k in sorted(_INDUSTRY_TO_SUPERSECTOR, key=len, reverse=True):
        if k in key:
            return _INDUSTRY_TO_SUPERSECTOR[k]
    return None


# ─── 3 · Helper: resolve Founding_Partner for a set of Person_IDs ─────────────

def _build_operator_lookup(
    mentor_cohort_df: pd.DataFrame,
    stakeholder_df: pd.DataFrame | None,
) -> pd.Series:
    """
    Returns a Series indexed by Person_ID → Founding_Partner (0/1).
    Priority: 04_MentorCohort → 02_Stakeholder_List fallback.
    Multiple cohort rows per mentor are resolved by taking the max (ever-operator).
    """
    op = (
        mentor_cohort_df[["Person_ID", "Founding_Partner"]]
        .copy()
        .assign(Person_ID=lambda x: x["Person_ID"].astype(str))
    )
    op["Founding_Partner"] = pd.to_numeric(op["Founding_Partner"], errors="coerce")
    op = op.sort_values("Cohort_Year") if "Cohort_Year" in op.columns else op
    op_lookup = op.groupby("Person_ID")["Founding_Partner"].last()

    if stakeholder_df is not None and "Founding_Partner" in stakeholder_df.columns:
        stk = (
            stakeholder_df[["Person_ID", "Founding_Partner"]]
            .copy()
            .assign(Person_ID=lambda x: x["Person_ID"].astype(str))
        )
        stk["Founding_Partner"] = pd.to_numeric(stk["Founding_Partner"], errors="coerce")
        stk_lookup = stk.set_index("Person_ID")["Founding_Partner"]
        op_lookup = op_lookup.combine_first(stk_lookup)

    return op_lookup


# ─── 4 · B1 — Mentor Participation Intensity ─────────────────────────────────

def _build_b1(
    sgm_df: pd.DataFrame,
    mentor_session_df: pd.DataFrame,
    mentor_cohort_df: pd.DataFrame,
    stakeholder_df: pd.DataFrame | None,
    venture_session_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (Venture_ID, Session_Num) with B1 features.

    Features:
        n_unique_mentors_sgm    — COUNT DISTINCT Person_ID per venture-session
        n_sgms_attended_total   — SUM of SGM events per venture-session
        avg_sgms_per_mentor     — depth ratio
        n_operator_mentors_sgm  — count of operators attending SGM
        pct_operator_sgm        — operator share
        avg_activeyrs_cdl_sgm   — mean CDL tenure across SGM mentors (session-level)
        pct_firsttime_mentors   — share of first-timers at the SGM
    """
    sgm = sgm_df.copy()
    sgm["Person_ID"] = sgm["Person_ID"].astype(str)
    sgm["Venture_ID"] = sgm["Venture_ID"].astype(str)

    grp = ["Venture_ID", "Session_Num"]

    n_unique = sgm.groupby(grp)["Person_ID"].nunique().rename("n_unique_mentors_sgm")
    n_total  = sgm.groupby(grp)["Person_ID"].count().rename("n_sgms_attended_total")

    b1 = pd.concat([n_unique, n_total], axis=1).reset_index()
    b1["avg_sgms_per_mentor"] = b1["n_sgms_attended_total"] / b1["n_unique_mentors_sgm"].replace(0, np.nan)

    # ── Operator features — join via 04_MentorCohort (cohort-matched preferred) ──
    op_lookup = _build_operator_lookup(mentor_cohort_df, stakeholder_df)
    sgm["is_operator"] = sgm["Person_ID"].map(op_lookup).fillna(0)

    op_counts = sgm.groupby(grp)["is_operator"].sum().rename("n_operator_mentors_sgm")
    b1 = b1.merge(op_counts.reset_index(), on=grp, how="left")
    b1["pct_operator_sgm"] = b1["n_operator_mentors_sgm"] / b1["n_unique_mentors_sgm"].replace(0, np.nan)

    # ── Experience features — use 06_MentorSession for session-level granularity ──
    ms = mentor_session_df.copy()
    ms["Person_ID"] = ms["Person_ID"].astype(str)

    # Build a lookup: (Person_ID, Cohort_Year, Session_Num) → N_ActiveYrs_CDL_Total
    if "N_ActiveYrs_CDL_Total" in ms.columns:
        activeyrs_col = "N_ActiveYrs_CDL_Total"
    elif "N_ActiveYrs_CDL" in ms.columns:
        activeyrs_col = "N_ActiveYrs_CDL"
    else:
        activeyrs_col = None

    if "Firsttime_Mentor" in ms.columns:
        firsttime_col = "Firsttime_Mentor"
    else:
        firsttime_col = None

    # Join SGM → Mentor_Session on (Person_ID, Session_Num, Cohort_Year / Cohort_ID)
    join_keys = ["Person_ID", "Session_Num"]
    for ck in ["Cohort_Year", "Cohort_ID"]:
        if ck in sgm.columns and ck in ms.columns:
            join_keys.append(ck)
            break

    ms_sub_cols = join_keys.copy()
    if activeyrs_col:
        ms_sub_cols.append(activeyrs_col)
    if firsttime_col:
        ms_sub_cols.append(firsttime_col)
    ms_sub = ms[ms_sub_cols].drop_duplicates()

    sgm_enriched = sgm.merge(ms_sub, on=join_keys, how="left")

    if activeyrs_col:
        avg_exp = (
            sgm_enriched.groupby(grp)[activeyrs_col]
            .mean()
            .rename("avg_activeyrs_cdl_sgm")
        )
        b1 = b1.merge(avg_exp.reset_index(), on=grp, how="left")
    else:
        b1["avg_activeyrs_cdl_sgm"] = np.nan

    if firsttime_col:
        sgm_enriched["is_firsttime"] = (
            pd.to_numeric(sgm_enriched[firsttime_col], errors="coerce").fillna(0)
        )
        pct_first = (
            sgm_enriched.groupby(grp)["is_firsttime"]
            .mean()
            .rename("pct_firsttime_mentors")
        )
        b1 = b1.merge(pct_first.reset_index(), on=grp, how="left")
    else:
        b1["pct_firsttime_mentors"] = np.nan

    return b1.set_index(grp)


# ─── 5 · B2 — Assigned Mentor Quality ────────────────────────────────────────

def _build_b2(
    venture_session_df: pd.DataFrame,
    mentor_cohort_df: pd.DataFrame,
    stakeholder_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (Venture_ID, Session_Num) with B2 features.

    Features:
        objsetter_is_operator   — 1 if obj-setter is operator
        objsetter_activeyrs     — N_ActiveYrs_CDL of obj-setter
        critique_is_operator    — 1 if critiquer is operator  (missing flag added)
        critique_activeyrs      — N_ActiveYrs_CDL of critiquer
    """
    vs = venture_session_df.copy()
    vs["Venture_ID"] = vs["Venture_ID"].astype(str)
    grp = ["Venture_ID", "Session_Num"]

    op_lookup = _build_operator_lookup(mentor_cohort_df, stakeholder_df)

    # Build Person_ID → N_ActiveYrs_CDL via mentor_cohort (take max across cohort rows)
    mc = mentor_cohort_df.copy()
    mc["Person_ID"] = mc["Person_ID"].astype(str)
    mc["N_ActiveYrs_CDL"] = pd.to_numeric(mc.get("N_ActiveYrs_CDL", pd.Series(dtype=float)), errors="coerce")
    mc = mc.sort_values("Cohort_Year")
    activeyrs_lookup = mc.groupby("Person_ID")["N_ActiveYrs_CDL"].last()

    b2 = vs[grp + ["ObjSetter_PersonID", "Critique_PersonID"]].copy()
    b2["ObjSetter_PersonID"] = pd.to_numeric(b2["ObjSetter_PersonID"], errors="coerce").astype("Int64").astype(str).replace("<NA>", np.nan)
    b2["Critique_PersonID"]  = pd.to_numeric(b2["Critique_PersonID"],  errors="coerce").astype("Int64").astype(str).replace("<NA>", np.nan)

    b2["objsetter_is_operator"] = b2["ObjSetter_PersonID"].map(op_lookup)
    b2["objsetter_activeyrs"]   = b2["ObjSetter_PersonID"].map(activeyrs_lookup)

    b2["critique_is_operator"] = b2["Critique_PersonID"].map(op_lookup)
    b2["critique_activeyrs"]   = b2["Critique_PersonID"].map(activeyrs_lookup)

    return b2[grp + [
        "objsetter_is_operator", "objsetter_activeyrs",
        "critique_is_operator", "critique_activeyrs",
    ]].set_index(grp)


# ─── 6 · B3 — Mentor–Venture Fit ─────────────────────────────────────────────

def _build_b3(
    sgm_df: pd.DataFrame,
    mentor_cohort_df: pd.DataFrame,
    stakeholder_df: pd.DataFrame | None,
    venture_supersector: pd.Series,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (Venture_ID, Session_Num) with B3 features.

    Features:
        n_domain_matched_mentors — count of SGM mentors whose industry maps to the
                                   same supersector as the venture
        pct_domain_matched       — n_domain_matched / n_unique_mentors_sgm
    """
    sgm = sgm_df.copy()
    sgm["Person_ID"]  = sgm["Person_ID"].astype(str)
    sgm["Venture_ID"] = sgm["Venture_ID"].astype(str)

    # Build mentor industry lookup (prefer 04_MentorCohort, fallback to 02_Stakeholder)
    mc = mentor_cohort_df.copy()
    mc["Person_ID"] = mc["Person_ID"].astype(str)

    if "Mentor_Industry" in mc.columns:
        ind_lookup = mc.groupby("Person_ID")["Mentor_Industry"].first()
    elif stakeholder_df is not None and "Mentor_Industry" in stakeholder_df.columns:
        stk = stakeholder_df.copy()
        stk["Person_ID"] = stk["Person_ID"].astype(str)
        ind_lookup = stk.groupby("Person_ID")["Mentor_Industry"].first()
    else:
        ind_lookup = pd.Series(dtype=str)

    sgm["mentor_supersector"] = sgm["Person_ID"].map(ind_lookup).apply(_mentor_industry_to_supersector)
    sgm["venture_supersector"] = sgm["Venture_ID"].map(venture_supersector)

    sgm["domain_match"] = (
        sgm["mentor_supersector"].notna()
        & sgm["venture_supersector"].notna()
        & (sgm["mentor_supersector"] == sgm["venture_supersector"])
    ).astype(int)

    grp = ["Venture_ID", "Session_Num"]
    n_match   = sgm.groupby(grp)["domain_match"].sum().rename("n_domain_matched_mentors")
    n_unique  = sgm.groupby(grp)["Person_ID"].nunique().rename("n_unique_mentors_sgm")

    b3 = pd.concat([n_match, n_unique], axis=1)
    b3["pct_domain_matched"] = b3["n_domain_matched_mentors"] / b3["n_unique_mentors_sgm"].replace(0, np.nan)

    return b3[["n_domain_matched_mentors", "pct_domain_matched"]]


# ─── 7 · B4 — Session Format & Competition ───────────────────────────────────

def _build_b4(
    venture_session_df: pd.DataFrame,
    mentor_session_df: pd.DataFrame,
) -> pd.DataFrame:

    vs = venture_session_df.copy()
    vs["Venture_ID"] = vs["Venture_ID"].astype(str)

    if "Session_Format" in vs.columns:
        vs["session_format"] = vs["Session_Format"].map(
            {"In person": 1, "In-person": 1, "Virtual": 0, 1: 1, 0: 0, "1": 1, "0": 0}
        )
    else:
        vs["session_format"] = np.nan

    vs["lrd_order"] = pd.to_numeric(vs.get("LRD_Order", pd.Series(dtype=float)), errors="coerce")

    status_col = None
    for c in ["Post_Session_Status", "Post_S1_Status"]:
        if c in vs.columns:
            status_col = c
            break

    session_keys = ["Session_Num", "Session_ID"]

    attending = vs[vs[status_col] != "DNA"] if status_col else vs
    n_ventures = (
        attending.groupby(session_keys)["Venture_ID"]
        .nunique()
        .rename("n_ventures_in_session")
    )

    ms = mentor_session_df.copy()
    ms["Person_ID"] = ms["Person_ID"].astype(str)
    n_mentors = (
        ms.groupby(session_keys)["Person_ID"]
        .nunique()
        .rename("n_mentors_in_session")
    )

    grp = ["Venture_ID", "Session_Num"]
    sel = list(dict.fromkeys(grp + ["Cohort_Year"] + session_keys))
    b4 = vs[[c for c in sel if c in vs.columns]].copy()
    b4["lrd_order"] = vs["lrd_order"].values
    b4["session_format"] = vs["session_format"].values

    merge_keys = [k for k in session_keys if k in b4.columns]
    b4 = b4.merge(n_ventures.reset_index(), on=merge_keys, how="left")
    b4 = b4.merge(n_mentors.reset_index(), on=merge_keys, how="left")

    b4["lrd_order_normalized"] = b4["lrd_order"] / b4["n_ventures_in_session"].replace(0, np.nan)
    b4["mentor_per_venture_ratio"] = b4["n_mentors_in_session"] / b4["n_ventures_in_session"].replace(0, np.nan)

    return b4[grp + ["Cohort_Year"] + [
    "session_format", "lrd_order", "lrd_order_normalized",
    "n_ventures_in_session", "mentor_per_venture_ratio",
    ]].set_index(grp)

# ─── 8 · Main builder ─────────────────────────────────────────────────────────

def build_block_B(
    sgm_df: pd.DataFrame,
    mentor_session_df: pd.DataFrame,
    mentor_cohort_df: pd.DataFrame,
    venture_session_df: pd.DataFrame,
    venture_cohort_df: pd.DataFrame,
    stakeholder_df: pd.DataFrame | None = None,
    miss_threshold: float = 0.50,
    imputer: IterativeImputer | None = None,
    fit_imputer: bool = True,
) -> tuple[pd.DataFrame, IterativeImputer, pd.DataFrame]:
    """
    Build Block B feature matrix.

    Parameters
    ----------
    sgm_df            : 08_SGM_Registrations — one row per mentor x venture x session
    mentor_session_df : 06_Mentor_Session   — one row per mentor x session
    mentor_cohort_df  : 04_Mentor_Cohort    — one row per mentor x cohort year
    venture_session_df: 05_VentureSession   — one row per venture x session
    venture_cohort_df : 03_Venture_Cohort   — used to pull NAICS supersector
    stakeholder_df    : 02_Stakeholder_List — optional fallback for mentor attributes
    miss_threshold    : columns with > this fraction missing are dropped (default 50%)
    imputer           : pre-fitted IterativeImputer; pass for val/test splits
    fit_imputer       : if True, fit imputer on this data (training split)

    Returns
    -------
    df        : engineered feature DataFrame (one row per Venture_ID x Session_Num)
    imputer   : fitted IterativeImputer (reuse on val/test)
    audit_df  : missingness audit table
    """

    # ── Venture supersector lookup (for B3 domain matching) ───────────────────
    vc = venture_cohort_df.copy()
    vc["Venture_ID"] = vc["Venture_ID"].astype(str)
    supersector_col = "NAICS2022_GPT_Supersector" if "NAICS2022_GPT_Supersector" in vc.columns else "NAICS2022_GPT_SSect"
    venture_supersector = vc.groupby("Venture_ID")[supersector_col].first() if supersector_col in vc.columns else pd.Series(dtype=str)

    # ── Anti-leakage filter: drop future cohort rows from mentor_cohort_df ──────
    max_cohort_year = venture_session_df["Cohort_Year"].max()
    mentor_cohort_df = mentor_cohort_df[
        mentor_cohort_df["Cohort_Year"] <= max_cohort_year
    ].reset_index(drop=True)

    # ── Build sub-blocks ───────────────────────────────────────────────────────
    b1 = _build_b1(sgm_df, mentor_session_df, mentor_cohort_df, stakeholder_df, venture_session_df)
    b2 = _build_b2(venture_session_df, mentor_cohort_df, stakeholder_df)
    b3 = _build_b3(sgm_df, mentor_cohort_df, stakeholder_df, venture_supersector)
    b4 = _build_b4(venture_session_df, mentor_session_df)

    # ── Join on (Venture_ID, Session_Num) ─────────────────────────────────────
    df = (
        b4
        .join(b1, how="left")
        .join(b2, how="left")
        .join(b3, how="left")
        .reset_index()
    )

    # ── Aggregate to venture x cohort_year x session granularity ─────────────
    # Duplicates arise when a venture appears in multiple cohorts: b2 is keyed on (Venture_ID, Session_Num) only, causing a cartesian product on join.
    GRP_KEYS = ["Venture_ID", "Cohort_Year", "Session_Num"]
    df = df.groupby(GRP_KEYS, as_index=False).mean(numeric_only=True)

    # ── Categorical imputation (mode-fill) ───────────────────────────────────
    CAT_MODE = []   # No categorical features in Block B

    if fit_imputer:
        cat_modes: dict[str, object] = {}
        if imputer is None:
            imputer = make_imputer()
        imputer._cat_modes_b = cat_modes
    else:
        cat_modes = imputer._cat_modes_b  # type: ignore[attr-defined]

    for col, mode_val in cat_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)

    # ── Missingness audit on numeric features ─────────────────────────────────
    NUMERIC_COLS = [
        # B1
        "n_unique_mentors_sgm", "n_sgms_attended_total", "avg_sgms_per_mentor",
        "n_operator_mentors_sgm", "pct_operator_sgm",
        "avg_activeyrs_cdl_sgm", "pct_firsttime_mentors",
        # B2
        "objsetter_is_operator", "objsetter_activeyrs",
        "critique_is_operator", "critique_activeyrs",
        # B3
        "n_domain_matched_mentors", "pct_domain_matched",
        # B4
        "session_format", "lrd_order", "lrd_order_normalized",
        "n_ventures_in_session", "mentor_per_venture_ratio",
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
        imputer._block_b_impute_cols = impute_cols
    else:
        impute_cols = imputer._block_b_impute_cols  # type: ignore[attr-defined]
        drop_cols = [c for c in NUMERIC_COLS if c in df.columns and c not in impute_cols]
        if drop_cols:
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        audit_df = pd.DataFrame()

    # ── BayesianRidge imputation ───────────────────────────────────────────────
    X_num = df[impute_cols].values

    if fit_imputer:
        X_imputed = imputer.fit_transform(X_num)
    else:
        X_imputed = imputer.transform(X_num)

    df[impute_cols] = X_imputed

    # Clip binary / proportion columns to [0, 1] and round binary flags
    BINARY_COLS = [
        c for c in impute_cols
        if any(kw in c for kw in ["binary", "is_operator", "session_format"])
    ]

    PROPORTION_COLS = [c for c in impute_cols if c.startswith("pct_") or c.endswith("_ratio") or c == "lrd_order_normalized"]
    df[BINARY_COLS] = df[BINARY_COLS].clip(0, 1).round()
    df[PROPORTION_COLS] = df[PROPORTION_COLS].clip(0, 1)

    # Ensure count columns are non-negative
    COUNT_COLS = [
        c for c in impute_cols
        if c.startswith("n_") or c in (
            "avg_sgms_per_mentor", "avg_activeyrs_cdl_sgm",
            "objsetter_activeyrs", "critique_activeyrs", "lrd_order"
        )
    ]
    df[COUNT_COLS] = df[COUNT_COLS].clip(lower=0).round()

    return df, imputer, audit_df

# ─── 9 · Quick smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n_v = 150   # ventures
    n_m = 80    # mentors
    n_s = 2     # sessions

    venture_ids  = [f"V{i:03d}" for i in range(n_v)]
    mentor_ids   = [f"M{i:03d}" for i in range(n_m)]
    cohort_years = ["2019/20", "2020/21", "2021/22", "2022/23", "2023/24"]

    # 08_SGM_Registrations - multiple mentors per venture-session
    rows = []
    for vid in venture_ids:
        for sess in range(1, n_s + 1):
            n_mentors_this = rng.integers(2, 8)
            chosen = rng.choice(mentor_ids, size=n_mentors_this, replace=False)
            yr = rng.choice(cohort_years)
            for mid in chosen:
                rows.append({"Venture_ID": vid, "Person_ID": mid,
                             "Session_Num": sess, "Cohort_Year": yr})
    sgm_df = pd.DataFrame(rows)

    # 04_Mentor_Cohort
    mentor_cohort_df = pd.DataFrame({
        "Person_ID":       mentor_ids,
        "Cohort_Year":     rng.choice(cohort_years, n_m),
        "N_ActiveYrs_CDL": rng.integers(1, 10, n_m).astype(float),
        "Firsttime_Mentor":rng.integers(0, 2, n_m).astype(float),
        "Founding_Partner": rng.integers(0, 2, n_m).astype(float),
        "Mentor_Industry": rng.choice(
            ["Healthcare", "Software", "Finance", "Manufacturing", None], n_m,
            p=[0.25, 0.30, 0.20, 0.15, 0.10]
        ),
    })

    # 06_Mentor_Session
    ms_rows = []
    for mid in mentor_ids:
        for sess in range(1, n_s + 1):
            ms_rows.append({
                "Person_ID": mid, "Session_Num": sess,
                "Cohort_Year": rng.choice(cohort_years),
                "N_ActiveYrs_CDL_Total": rng.integers(1, 10),
                "Firsttime_Mentor": rng.integers(0, 2),
                "N_SGMs_Attended": rng.integers(0, 5),
                "N_Hands_Ment": rng.integers(0, 3),
            })
    mentor_session_df = pd.DataFrame(ms_rows)

    # 05_VentureSession
    vs_rows = []
    for vid in venture_ids:
        for sess in range(1, n_s + 1):
            yr = rng.choice(cohort_years)
            vs_rows.append({
                "Venture_ID": vid, "Session_Num": sess,
                "Cohort_Year": yr,
                "Post_Session_Status": rng.choice(["Passed", "Dropped", "DNA"], p=[0.6, 0.3, 0.1]),
                "LRD_Order": rng.integers(1, 25) if rng.random() > 0.15 else None,
                "Session_Format": rng.choice(["In person", "Virtual"]),
                "ObjSetter_PersonID": rng.choice(mentor_ids),
                "Critique_PersonID":  rng.choice(mentor_ids) if rng.random() > 0.2 else None,
                "Hands_Mentorship": rng.integers(0, 4),
            })
    venture_session_df = pd.DataFrame(vs_rows)

    # 03_VentureCohort - minimal for supersector
    supersectors = [
        "1_Goods Producing Industries", "3_Information",
        "4_Financial Activities", "6_Education and Health Services",
    ]
    venture_cohort_df = pd.DataFrame({
        "Venture_ID":              venture_ids,
        "Cohort_Year":             rng.choice(cohort_years, n_v),
        "NAICS2022_GPT_Supersector": rng.choice(supersectors + [None], n_v, p=[0.2, 0.3, 0.2, 0.2, 0.1]),
    })

    df_out, fitted_imputer, audit = build_block_B(
        sgm_df=sgm_df,
        mentor_session_df=mentor_session_df,
        mentor_cohort_df=mentor_cohort_df,
        venture_session_df=venture_session_df,
        venture_cohort_df=venture_cohort_df,
        stakeholder_df=None,
        miss_threshold=0.50,
    )

    print(f"\nOutput shape : {df_out.shape}")
    print(df_out.head(10).to_string())
    print("\nRemaining NaNs per column:")
    nans = df_out.isnull().sum()
    print(nans[nans > 0] if nans.any() else "  → none")