"""
Assembles the master modeling dataset by merging Block A–E feature DataFrames with the binary target variable, then pivots to venture x cohort granularity.

Pipeline:
  1. build_master_dataset() → venture x session (intermediate)
  2. pivot_to_venture_level() → venture x cohort (final)

Output schema (one row = one venture x one cohort):
  - venture_id
  - cohort_year
  - [block_a features]          ← static, same for S1 and S2
  - [block_b/c/d/e features_s1] ← S1 features
  - [block_b/c/d/e features_s2] ← S2 features (NaN if dropped after S1)
  - target ← 1 if venture receives ≥1 mentorship hand in S2, else 0 (0 if dropped after S1, i.e. no S2 row)
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─── Keys ─────────────────────────────────────────────────────────────────────

VENTURE_ID   = "Venture_ID"
COHORT_YEAR  = "Cohort_Year"
SESSION_NUM  = "Session_Num"

MERGE_KEYS   = [VENTURE_ID, COHORT_YEAR, SESSION_NUM]
INDEX_COLS   = [VENTURE_ID, COHORT_YEAR, SESSION_NUM]

VALID_SESSIONS = {1, 2}


# ─── 1 · Build binary target ──────────────────────────────────────────────────

def build_target(venture_session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the session-level binary target from 05_Venture_Session.
    target = 1 if Hands_Mentorship > 0, else 0.
    Used internally by build_master_dataset.
    The final venture-level target is built in pivot_to_venture_level.
    """
    vs = (
        venture_session_df[
            venture_session_df[SESSION_NUM].isin(VALID_SESSIONS)
        ][[VENTURE_ID, COHORT_YEAR, SESSION_NUM, "Hands_Mentorship"]]
        .drop_duplicates(subset=[VENTURE_ID, COHORT_YEAR, SESSION_NUM])
        .copy()
    )

    n_before = len(vs)
    vs = vs.dropna(subset=["Hands_Mentorship"])
    n_dropped = n_before - len(vs)
    if n_dropped > 0:
        print(f"  ⚠ Dropped {n_dropped} rows with missing Hands_Mentorship.")

    vs["target"] = (vs["Hands_Mentorship"] > 0).astype(int)
    return vs[[VENTURE_ID, COHORT_YEAR, SESSION_NUM, "target"]]


# ─── 2 · Validate block shape ─────────────────────────────────────────────────

def _validate_block(df: pd.DataFrame, block_name: str) -> None:
    missing_keys = [k for k in MERGE_KEYS if k not in df.columns]
    if missing_keys:
        raise ValueError(
            f"Block {block_name} is missing required columns: {missing_keys}. "
            f"Found: {df.columns.tolist()}"
        )
    dupes = df.duplicated(subset=MERGE_KEYS).sum()
    if dupes > 0:
        raise ValueError(
            f"Block {block_name} has {dupes} duplicate rows on {MERGE_KEYS}."
        )
    invalid_sessions = set(df[SESSION_NUM].unique()) - VALID_SESSIONS
    if invalid_sessions:
        raise ValueError(
            f"Block {block_name} contains unexpected session numbers: {invalid_sessions}."
        )


# ─── 3 · Block A expander ─────────────────────────────────────────────────────

def expand_block_a(df_a: pd.DataFrame) -> pd.DataFrame:
    """
    Block A contains static venture-level data (one row per venture).
    Duplicate each row for Session_Num = 1 and Session_Num = 2 so it can
    be merged at venture x session granularity.
    """
    if SESSION_NUM in df_a.columns:
        return df_a  # already expanded
    s1 = df_a.copy()
    s1[SESSION_NUM] = 1
    s2 = df_a.copy()
    s2[SESSION_NUM] = 2
    return pd.concat([s1, s2], ignore_index=True)


# ─── 4 · Prefix block columns ─────────────────────────────────────────────────

def _prefix_block_cols(df: pd.DataFrame, block_name: str) -> pd.DataFrame:
    prefix  = block_name.lower() + "_"
    key_set = set(INDEX_COLS)
    rename_map = {
        col: prefix + col
        for col in df.columns
        if col not in key_set
    }
    return df.rename(columns=rename_map)


# ─── 5 · Master merge (venture × session) ─────────────────────────────────────

def build_master_dataset(
    blocks: dict[str, pd.DataFrame],
    venture_session_df: pd.DataFrame,
    prefix_columns: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge all feature blocks with the session-level target.
    Output is at venture x session granularity (intermediate step).
    Call pivot_to_venture_level() afterwards to get the final dataset.
    """
    if verbose:
        print("Building target variable...")
    target_df = build_target(venture_session_df)
    if verbose:
        n_pos = target_df["target"].sum()
        n_tot = len(target_df)
        print(f"  Universe    : {n_tot} venture-session pairs")
        print(f"  Positive (1): {n_pos} ({100 * n_pos / n_tot:.1f}%)")
        print(f"  Negative (0): {n_tot - n_pos} ({100 * (1 - n_pos / n_tot):.1f}%)")

    master = target_df.copy()

    for block_name, block_df in blocks.items():
        if verbose:
            n_feat = block_df.shape[1] - len(MERGE_KEYS)
            print(f"\nMerging Block {block_name}  ({n_feat} features, {len(block_df)} rows)...")

        _validate_block(block_df, block_name)

        if prefix_columns:
            block_df = _prefix_block_cols(block_df, block_name)

        before = len(master)
        master = master.merge(block_df, on=MERGE_KEYS, how="left")
        after  = len(master)

        if before != after:
            raise RuntimeError(
                f"Row count changed after merging Block {block_name}: "
                f"{before} → {after}. Check for duplicate merge keys."
            )

        if verbose:
            n_missing = master.isnull().any(axis=1).sum()
            print(f"  ✓ Merged. Rows with ≥1 NaN: {n_missing} / {len(master)}")

    master = master.rename(columns={
        VENTURE_ID:  "venture_id",
        COHORT_YEAR: "cohort_year",
        SESSION_NUM: "session_number",
    })

    key_cols     = ["venture_id", "cohort_year", "session_number"]
    feature_cols = [c for c in master.columns if c not in key_cols + ["target"]]
    master       = master[key_cols + feature_cols + ["target"]]

    if verbose:
        print("\n" + "=" * 60)
        print("MASTER DATASET (venture × session) READY")
        print("=" * 60)
        print(f"  Shape            : {master.shape[0]} rows × {master.shape[1]} cols")
        print(f"  Feature columns  : {len(feature_cols)}")
        for sn, grp in master.groupby("session_number"):
            n_pos = grp["target"].sum()
            print(f"    S{sn}: {len(grp)} rows | target=1: {n_pos} ({100*n_pos/len(grp):.1f}%)")
        total_missing = master[feature_cols].isnull().sum().sum()
        total_cells   = master[feature_cols].shape[0] * master[feature_cols].shape[1]
        print(f"  Overall NaN rate : {100 * total_missing / total_cells:.2f}%")
        print("=" * 60)

    return master


# ─── 6 · Pivot to venture × cohort ────────────────────────────────────────────

def pivot_to_venture_level(
    master: pd.DataFrame,
    static_block: str = "A",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Transform the venture x session dataset into a venture xcohort dataset.

    Parameters
    ----------
    master       : output of build_master_dataset()
    static_block : block letter whose features are static (default "A")
    verbose      : print diagnostics

    Returns
    -------
    DataFrame at venture x cohort granularity.
    """
    prefix_static = static_block.lower() + "_"

    all_feature_cols = [
        c for c in master.columns
        if c not in ["venture_id", "cohort_year", "session_number", "target"]
    ]
    static_cols  = [c for c in all_feature_cols if c.startswith(prefix_static)]
    dynamic_cols = [c for c in all_feature_cols if not c.startswith(prefix_static)]

    # ── Separate S1 and S2 ────────────────────────────────────────────────────
    s1 = master[master["session_number"] == 1].copy()
    s2 = master[master["session_number"] == 2].copy()

    # Drop S2-only ventures (ambiguous — no S1 row)
    s1_ids  = set(zip(s1["venture_id"], s1["cohort_year"]))
    s2_ids  = set(zip(s2["venture_id"], s2["cohort_year"]))
    s2_only = s2_ids - s1_ids

    if s2_only and verbose:
        print(f"  ⚠ Dropping {len(s2_only)} ventures with S2 but no S1 (ambiguous).")
    if s2_only:
        s2 = s2[~s2.apply(lambda r: (r["venture_id"], r["cohort_year"]) in s2_only, axis=1)]

    # ── Start from S1 universe ─────────────────────────────────────────────────
    # All ventures with S1 are included - dropped ones get target = 0
    venture_level = s1[["venture_id", "cohort_year"] + static_cols].copy()

    # ── Final target: S2 hand raise ───────────────────────────────────────────
    s2_target = s2[["venture_id", "cohort_year", "target"]].rename(
        columns={"target": "target_s2"}
    )
    venture_level = venture_level.merge(
        s2_target, on=["venture_id", "cohort_year"], how="left"
    )
    # Ventures dropped after S1 -> no S2 row -> target = 0
    venture_level["target"] = venture_level["target_s2"].fillna(0).astype(int)
    venture_level.drop(columns=["target_s2"], inplace=True)

    # ── S1 dynamic features ───────────────────────────────────────────────────
    s1_dynamic = s1[["venture_id", "cohort_year"] + dynamic_cols].rename(
        columns={c: c + "_s1" for c in dynamic_cols}
    )
    venture_level = venture_level.merge(
        s1_dynamic, on=["venture_id", "cohort_year"], how="left"
    )

    # ── S2 dynamic features ───────────────────────────────────────────────────
    s2_dynamic = s2[["venture_id", "cohort_year"] + dynamic_cols].rename(
        columns={c: c + "_s2" for c in dynamic_cols}
    )
    venture_level = venture_level.merge(
        s2_dynamic, on=["venture_id", "cohort_year"], how="left"
    )
    # Ventures dropped after S1 -> S2 features = NaN (correct - no S2 data)

    # ── Ajouter indicatrice reached_s2 et imputer features S2 à 0 ────────────
    s2_cols = [c for c in venture_level.columns if c.endswith("_s2")]
    venture_level["reached_s2"] = venture_level[s2_cols[0]].notna().astype(int) if s2_cols else 0
    venture_level[s2_cols] = venture_level[s2_cols].fillna(0)

    # ── Reorder columns ───────────────────────────────────────────────────────
    key_cols      = ["venture_id", "cohort_year"]
    feature_cols  = [c for c in venture_level.columns if c not in key_cols + ["target"]]
    venture_level = venture_level[key_cols + feature_cols + ["target"]]
    venture_level = venture_level.reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 60)
        print("FINAL DATASET (venture × cohort) READY")
        print("=" * 60)
        n_pos = venture_level["target"].sum()
        n_tot = len(venture_level)
        print(f"  Shape            : {n_tot} ventures × {venture_level.shape[1]} cols")
        print(f"  Feature columns  : {len(feature_cols)}")
        print(f"  Positive (1)     : {n_pos} ({100 * n_pos / n_tot:.1f}%)")
        print(f"  Negative (0)     : {n_tot - n_pos} ({100 * (1 - n_pos / n_tot):.1f}%)")
        total_missing = venture_level[feature_cols].isnull().sum().sum()
        total_cells   = venture_level[feature_cols].shape[0] * venture_level[feature_cols].shape[1]
        print(f"  Overall NaN rate : {100 * total_missing / total_cells:.2f}%")
        print("=" * 60)

    return venture_level


# ─── 7 · Save ─────────────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, path: str, fmt: str = "csv") -> None:
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "excel":
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}.")
    print(f"Saved → {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")


# ─── 8 · Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N_VENTURES   = 80
    COHORT_YEARS = ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]

    venture_ids = [f"V{i:04d}" for i in range(N_VENTURES)]
    cohort_map  = {v: rng.choice(COHORT_YEARS) for v in venture_ids}

    vs_rows = []
    for v in venture_ids:
        vs_rows.append({
            "Venture_ID": v, "Cohort_Year": cohort_map[v], "Session_Num": 1,
            "Hands_Mentorship": rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1]),
        })
        if rng.random() > 0.3:  # ~70% make it to S2
            vs_rows.append({
                "Venture_ID": v, "Cohort_Year": cohort_map[v], "Session_Num": 2,
                "Hands_Mentorship": rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1]),
            })
    venture_session_df = pd.DataFrame(vs_rows)

    # Block A — static, no Session_Num
    df_a = venture_session_df[["Venture_ID", "Cohort_Year"]].drop_duplicates().copy()
    for i in range(5):
        df_a[f"feature_{i}"] = rng.standard_normal(len(df_a))

    def _fake_block(n_features):
        df = venture_session_df[["Venture_ID", "Cohort_Year", "Session_Num"]].copy()
        for i in range(n_features):
            col = f"feature_{i}"
            df[col] = rng.standard_normal(len(df))
            df.loc[rng.random(len(df)) < 0.05, col] = np.nan
        return df

    blocks = {
        "A": expand_block_a(df_a),
        "B": _fake_block(6),
        "C": _fake_block(5),
        "D": _fake_block(8),
        "E": _fake_block(4),
    }

    master = build_master_dataset(blocks=blocks, venture_session_df=venture_session_df)
    final  = pivot_to_venture_level(master, static_block="A")

    print("\nFirst 3 rows:")
    print(final.head(3).to_string())
    save_dataset(final, "/tmp/test_dataset.csv", fmt="csv")