"""
Block C — LRD Discussion Features
Preprocessing with:
  1. Missingness audit     
  2. BayesianRidge imputation 
  3. Full spec alignment  
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

try:
    from transformers import pipeline as hf_pipeline
    _FINBERT_AVAILABLE = True
except ImportError:
    _FINBERT_AVAILABLE = False

warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", message=".*truncation.*")

_FINBERT_SCORE = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}


# --- Keyword lexicons (C2) ---------------------------------------------------

_OBJECTION_KEYWORDS = [
    "concern", "risk", "not sure", "unclear", "worried", "doubt",
    "uncertain", "skeptic", "challeng", "problem", "issue", "weak",
    "struggle", "difficult", "not convinced", "don't see", "barrier",
]

_ENDORSEMENT_KEYWORDS = [
    "great", "excellent", "impressive", "strong", "love", "excited",
    "enthusiastic", "compelling", "believe in", "support", "potential",
    "opportunity", "promising", "outstanding", "recommend", "definitely",
]

# Canonical group keys used everywhere
_GRP = ["Venture_ID", "Session_Num", "Cohort_Year"]


# --- 0 . Missingness audit ---------------------------------------------------

def audit_missingness(
    df: pd.DataFrame,
    threshold: float = 0.50,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rates = df.isnull().mean().sort_values(ascending=False)
    audit_df = pd.DataFrame({
        "feature":     rates.index,
        "n_missing":   df.isnull().sum().reindex(rates.index).values,
        "pct_missing": rates.values,
    })
    audit_df["decision"] = np.where(
        audit_df["pct_missing"] > threshold, "DROP", "KEEP"
    )
    if verbose:
        print(f"\n-- Block C missingness audit (threshold={threshold:.0%}) --")
        print(audit_df.to_string(index=False))
        n_drop = (audit_df["decision"] == "DROP").sum()
        print(f"\n  -> Dropping {n_drop} / {len(audit_df)} columns\n")
    keep = audit_df.loc[audit_df["decision"] == "KEEP", "feature"].tolist()
    drop = audit_df.loc[audit_df["decision"] == "DROP", "feature"].tolist()
    return audit_df, keep, drop


# --- C1 . Participation Metrics ----------------------------------------------

def _build_c1(
    notes_df: pd.DataFrame,
    mentor_session_df: pd.DataFrame,
) -> pd.DataFrame:
    df = notes_df.copy()
    FOUNDER_SPEAKER_IDS = {229259, 107505}
    df["Is_Founder"] = df["Speaker_ID"].isin(FOUNDER_SPEAKER_IDS).astype(int)

    grp = df.groupby(_GRP)

    n_unique_speakers = grp["Speaker_ID"].nunique().rename("n_unique_speakers")
    n_total_comments  = grp.size().rename("n_total_comments")

    mentor_mask = df["Is_Founder"] == 0
    n_mentor_comments = (
        df[mentor_mask]
        .groupby(_GRP)["Comment_Num"]
        .count()
        .rename("n_mentor_comments")
    )

    founder_mask = df["Is_Founder"] == 1
    n_founder_comments = (
        df[founder_mask]
        .groupby(_GRP)["Comment_Num"]
        .count()
        .rename("n_founder_comments")
    )

    c1 = pd.concat(
        [n_unique_speakers, n_total_comments,
         n_mentor_comments, n_founder_comments],
        axis=1,
    ).reset_index()

    c1["founder_comment_ratio"] = (
        c1["n_founder_comments"] / c1["n_total_comments"].replace(0, np.nan)
    )

    # N_Mentors_In_Session - COUNT DISTINCT Person_ID per Session_ID
    # then map back to Venture_ID + Session_Num + Cohort_Year via notes_df
    n_mentors = (
        mentor_session_df.groupby("Session_ID")["Person_ID"]
        .nunique()
        .rename("N_Mentors_In_Session")
        .reset_index()
    )

    # Get the Session_ID -> Venture_ID + Session_Num + Cohort_Year mapping
    session_map = (
        df[["Venture_ID", "Session_Num", "Cohort_Year", "Session_ID"]]
        .drop_duplicates()
    )

    c1 = c1.merge(session_map, on=_GRP, how="left")
    c1 = c1.merge(n_mentors, on="Session_ID", how="left")

    c1["silence_score"] = (
        c1["n_unique_speakers"]
        / c1["N_Mentors_In_Session"].replace(0, np.nan)
    )
    c1.drop(columns=["N_Mentors_In_Session", "Session_ID"], inplace=True)

    return c1


# --- C2 . Objection & Criticism Metrics -------------------------------------

def _classify_comment(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str) or text.strip() == "":
        return 0
    text_lower = text.lower()
    return int(any(kw in text_lower for kw in keywords))


def _build_c2(notes_df: pd.DataFrame) -> pd.DataFrame:
    df = notes_df.copy()

    df["_has_question"]   = df["Comment"].str.contains(r"\?", na=False).astype(int)
    df["_is_objection"]   = df["Comment"].apply(_classify_comment, keywords=_OBJECTION_KEYWORDS)
    df["_is_endorsement"] = df["Comment"].apply(_classify_comment, keywords=_ENDORSEMENT_KEYWORDS)

    grp = df.groupby(_GRP)
    total         = grp.size().rename("_total")
    question_n    = grp["_has_question"].sum().rename("_q")
    objection_n   = grp["_is_objection"].sum().rename("_obj")
    endorsement_n = grp["_is_endorsement"].sum().rename("_end")

    c2 = pd.concat([total, question_n, objection_n, endorsement_n], axis=1).reset_index()

    c2["question_density"]  = c2["_q"]   / c2["_total"].replace(0, np.nan)
    c2["objection_ratio"]   = c2["_obj"] / c2["_total"].replace(0, np.nan)
    c2["endorsement_ratio"] = c2["_end"] / c2["_total"].replace(0, np.nan)

    c2.drop(columns=["_total", "_q", "_obj", "_end"], inplace=True)
    return c2


# --- C3 . NLP Signals --------------------------------------------------------

_FINBERT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def _load_finbert():
    if not _FINBERT_AVAILABLE:
        return None
    return hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",  
        truncation=True,
        max_length=512,
    )


def _score_with_finbert(texts: list[str], pipe) -> list[float]:
    valid_mask  = [isinstance(t, str) and t.strip() != "" for t in texts]
    valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]
    scores = [np.nan] * len(texts)
    if not valid_texts:
        return scores
    results    = pipe(valid_texts, batch_size=32)
    valid_iter = iter(results)
    for i, ok in enumerate(valid_mask):
        if ok:
            res = next(valid_iter)
            scores[i] = _FINBERT_SCORE.get(res["label"], 0.0)
    return scores


def _score_fallback(texts: list[str]) -> list[float]:
    scores = []
    for text in texts:
        if not isinstance(text, str) or text.strip() == "":
            scores.append(np.nan)
            continue
        pos   = sum(text.lower().count(kw) for kw in _ENDORSEMENT_KEYWORDS)
        neg   = sum(text.lower().count(kw) for kw in _OBJECTION_KEYWORDS)
        total = pos + neg
        scores.append(0.0 if total == 0 else (pos - neg) / total)
    return scores


def _specificity_score(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    has_number       = bool(re.search(r"\d+", text))
    has_date         = bool(re.search(
        r"\b(\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\w*\b|\bQ[1-4]\b)", text, re.IGNORECASE,
    ))
    has_named_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))
    return float(has_number or has_date or has_named_entity)


def _build_c3(notes_df: pd.DataFrame) -> pd.DataFrame:
    df = notes_df.copy()
    FOUNDER_SPEAKER_IDS = {229259, 107505}
    df["Is_Founder"] = df["Speaker_ID"].isin(FOUNDER_SPEAKER_IDS).astype(int)

    # -- Sentiment via FinBERT (or fallback) ----------------------------------
    pipe  = _load_finbert()
    texts = df["Comment"].tolist()

    if pipe is not None:
        print("  [C3] Running FinBERT sentiment scoring...")
        df["_sent"] = _score_with_finbert(texts, pipe)
    else:
        print("  [C3] transformers not found - using keyword fallback for sentiment.")
        df["_sent"] = _score_fallback(texts)

    mentor_df = df[df["Is_Founder"] == 0].copy()
    sentiment_score = (
        mentor_df.groupby(_GRP)["_sent"]
        .mean()
        .rename("sentiment_score")
    )

    # -- Supportiveness index -------------------------------------------------
    df["_is_endorse"] = df["Comment"].apply(_classify_comment, keywords=_ENDORSEMENT_KEYWORDS)
    df["_is_object"]  = df["Comment"].apply(_classify_comment, keywords=_OBJECTION_KEYWORDS)
    mentor_df = df[df["Is_Founder"] == 0].copy()

    def _supportiveness(sub_df: pd.DataFrame) -> float:
        e     = sub_df["_is_endorse"].sum()
        o     = sub_df["_is_object"].sum()
        total = e + o
        return np.nan if total == 0 else (e - o) / total

    supportiveness_index = (
        mentor_df.groupby(_GRP)
        .apply(_supportiveness)
        .rename("supportiveness_index")
    )

    # -- Specificity score ----------------------------------------------------
    df["_specific"] = df["Comment"].apply(_specificity_score)
    specificity_score = (
        df.groupby(_GRP)["_specific"]
        .mean()
        .rename("specificity_score")
    )

    c3 = pd.concat(
        [sentiment_score, supportiveness_index, specificity_score], axis=1
    ).reset_index()

    return c3


# --- Master builder ----------------------------------------------------------

def build_block_C(
    notes_df: pd.DataFrame,
    mentor_session_df: pd.DataFrame,
    fit_imputer: bool = True,
    imputer: Optional[IterativeImputer] = None,
    missingness_threshold: float = 0.50,
    verbose: bool = True,
) -> tuple[pd.DataFrame, IterativeImputer, pd.DataFrame]:
    """
    Full Block C pipeline.

    Parameters
    ----------
    notes_df              : 10_Meeting_Notes table.
                            Must contain Cohort_Year — join from venture_session
                            before calling if absent.
    mentor_session_df     : 06_Mentor_Session table (for N_Mentors_In_Session).
    fit_imputer           : True on train split; False on val / test splits.
    imputer               : Pre-fitted imputer to pass in when fit_imputer=False.
    missingness_threshold : Drop columns above this missingness rate.
    verbose               : Print audit table.

    Returns
    -------
    df_out   : Feature DataFrame at venture x session x cohort granularity.
    imputer  : Fitted IterativeImputer (reuse on val/test).
    audit_df : Missingness audit table.
    """
    # Guard — Cohort_Year must be present
    if "Cohort_Year" not in notes_df.columns:
        raise ValueError(
            "notes_df must contain 'Cohort_Year'. "
            "Join it from venture_session before calling build_block_C."
        )

    # 1 . Build sub-blocks
    c1 = _build_c1(notes_df, mentor_session_df)
    c2 = _build_c2(notes_df)
    c3 = _build_c3(notes_df)

    # 2 . Merge into one venture × session × cohort frame
    df = c1.merge(c2, on=_GRP, how="outer")
    df = df.merge(c3, on=_GRP, how="outer")

    # Aggregate to correct granularity: duplicates arise when a venture maps
    # to multiple Session_IDs in _build_c1, producing different silence_score.
    df = df.groupby(_GRP, as_index=False).mean(numeric_only=True)

    # 3 . Missingness audit
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols      = ["Venture_ID", "Session_Num", "Cohort_Year"]
    feature_cols = [c for c in numeric_cols if c not in id_cols]

    audit_df, keep_cols, drop_cols = audit_missingness(
        df[feature_cols], threshold=missingness_threshold, verbose=verbose,
    )

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        keep_feature_cols = [c for c in feature_cols if c in keep_cols]
    else:
        keep_feature_cols = feature_cols

    # 4 . BayesianRidge imputation
    impute_cols = keep_feature_cols

    if fit_imputer:
        imputer = IterativeImputer(
            estimator=BayesianRidge(), max_iter=10, random_state=42, verbose=0,
        )
        df[impute_cols] = imputer.fit_transform(df[impute_cols])
        imputer._block_c_impute_cols = impute_cols  # type: ignore[attr-defined]
    else:
        if imputer is None:
            raise ValueError("fit_imputer=False requires a pre-fitted imputer.")
        impute_cols = imputer._block_c_impute_cols  # type: ignore[attr-defined]
        for col in impute_cols:
            if col not in df.columns:
                df[col] = np.nan
        df[impute_cols] = imputer.transform(df[impute_cols])

    # 5 . Post-imputation clipping
    RATIO_COLS     = [
        "founder_comment_ratio", "silence_score", "question_density",
        "objection_ratio", "endorsement_ratio", "specificity_score",
    ]
    SENTIMENT_COLS = ["sentiment_score", "supportiveness_index"]
    COUNT_COLS     = [
        "n_unique_speakers", "n_total_comments",
        "n_mentor_comments", "n_founder_comments",
    ]

    for col in RATIO_COLS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)
    for col in SENTIMENT_COLS:
        if col in df.columns:
            df[col] = df[col].clip(-1.0, 1.0)
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0).round().astype("Int64")

    return df, imputer, audit_df


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N_VENTURES   = 60
    venture_ids  = [f"V{i:03d}" for i in range(N_VENTURES)]
    sessions     = [1, 2]
    cohort_years = ["2019/20", "2020/21", "2021/22", "2022/23"]

    sample_texts = [
        "I'm concerned about the market risk here.",
        "This is really impressive work, strong potential.",
        "Not sure about the go-to-market strategy.",
        "Excellent traction, I support this venture.",
        "What's your revenue target for Q4 2025?",
        "I doubt they can scale without more funding.",
        "Great team composition and technical depth.",
        "",
        "The market size assumption is unclear.",
        "We raised 2M CAD in pre-seed already.",
    ]

    # Assign each venture a cohort year
    venture_cohort = {vid: rng.choice(cohort_years) for vid in venture_ids}

    comments = []
    for vid in venture_ids:
        for snum in sessions:
            n = rng.integers(3, 9)
            sid = f"SID_{vid}_{snum}"
            for j in range(n):
                is_founder = (j == 0)
                comments.append({
                    "Venture_ID":  vid,
                    "Session_Num": snum,
                    "Cohort_Year": venture_cohort[vid],
                    "Session_ID":  sid,
                    "Comment_Num": j + 1,
                    "Speaker_ID":  None if is_founder else f"SP_{rng.integers(1, 20):02d}",
                    "Comment":     rng.choice(sample_texts),
                })

    notes_df = pd.DataFrame(comments)

    mentor_session_rows = []
    for vid in venture_ids:
        for snum in sessions:
            sid = f"SID_{vid}_{snum}"
            mentor_session_rows.append({
                "Session_ID":  sid,
                "Person_ID":   f"M_{rng.integers(1, 50):02d}",
                "Cohort_Year": venture_cohort[vid],
            })
    mentor_session_df = pd.DataFrame(mentor_session_rows)

    CUTOFF = "2021/22"
    train_ventures = [v for v, cy in venture_cohort.items() if cy <= CUTOFF]
    test_ventures  = [v for v, cy in venture_cohort.items() if cy >  CUTOFF]

    notes_train = notes_df[notes_df["Venture_ID"].isin(train_ventures)]
    notes_test  = notes_df[notes_df["Venture_ID"].isin(test_ventures)]
    ms_train    = mentor_session_df[mentor_session_df["Cohort_Year"] <= CUTOFF]
    ms_test     = mentor_session_df[mentor_session_df["Cohort_Year"] >  CUTOFF]

    print("=" * 60)
    print("Block C smoke test")
    print("=" * 60)

    df_train, fitted_imputer, audit = build_block_C(
        notes_train, ms_train, fit_imputer=True, verbose=True
    )
    df_test, _, _ = build_block_C(
        notes_test, ms_test,
        fit_imputer=False, imputer=fitted_imputer, verbose=False,
    )

    print(f"Train shape : {df_train.shape}")
    print(f"Test  shape : {df_test.shape}")
    print(f"Train NaNs  : {df_train.isnull().sum().sum()}")
    print(f"Test  NaNs  : {df_test.isnull().sum().sum()}")
    print("\nTrain head:")
    print(df_train.head())
    print("\nCohort_Year in output:", "Cohort_Year" in df_train.columns)