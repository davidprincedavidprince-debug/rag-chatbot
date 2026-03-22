"""
pre_aggregate.py

Condenses scoring_dashboard.xlsx from ~14,000 raw response rows
into ~200-300 meaningful summary rows that the chatbot can reason over.

Run once from your project folder:
    python pre_aggregate.py

Output: data/scoring_summary.xlsx  (replaces scoring_dashboard.xlsx in the index)
"""

import pandas as pd
import numpy as np
import os

INPUT_FILE  = "data/scoring_dashboard.xlsx"
OUTPUT_FILE = "data/scoring_summary.xlsx"

SCORE_COLS = [
    "Relevance_WA_Priority",
    "Relevance_WA_Sufficiency",
    "Relevance_overall",
    "Efficiency_WA_Timeliness",
    "Efficiency_WA_Quality",
    "Efficiency_overall",
    "Sustainability_WA_Measures_Sustainability",
    "Sustainability_WA_Current_Status",
    "Sustainability_overall",
]

IMPACT_COLS        = ["Impact_WA", "Impact_total_responses"]
EFFECTIVENESS_COLS = ["Effectiveness_WA", "Effectiveness_total_responses"]


def safe_score_cols(df, cols):
    """Return only the score columns that exist in this sheet."""
    return [c for c in cols if c in df.columns]


def round_scores(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df


# ── Sheet: L4 — Tool level ────────────────────────────────────────────
def aggregate_l4(df: pd.DataFrame) -> pd.DataFrame:
    """One row per project_code + Tool_level4 with mean scores."""
    score_cols = safe_score_cols(df, SCORE_COLS + IMPACT_COLS + EFFECTIVENESS_COLS)
    agg        = {c: "mean" for c in score_cols}
    agg["project_code"] = "first"

    result = (
        df.groupby(["project_code", "Tool_level4"], as_index=False)
          .agg(agg)
    )
    result = round_scores(result, score_cols)
    result["summary_level"] = "L4 - Tool"
    print(f"  L4: {len(df)} rows → {len(result)} aggregated rows")
    return result


# ── Sheet: L3 — Intervention ──────────────────────────────────────────
def aggregate_l3(df: pd.DataFrame) -> pd.DataFrame:
    """One row per project + tool + intervention."""
    score_cols  = safe_score_cols(df, SCORE_COLS + IMPACT_COLS + EFFECTIVENESS_COLS)
    group_cols  = ["project_code", "Tool_level4", "Intervention_level3"]
    group_cols  = [c for c in group_cols if c in df.columns]
    agg         = {c: "mean" for c in score_cols}

    result = df.groupby(group_cols, as_index=False).agg(agg)
    result = round_scores(result, score_cols)
    result["summary_level"] = "L3 - Intervention"
    print(f"  L3: {len(df)} rows → {len(result)} aggregated rows")
    return result


# ── Sheet: L2 — Intervention Category ────────────────────────────────
def aggregate_l2(df: pd.DataFrame) -> pd.DataFrame:
    """One row per project + tool + intervention + category."""
    score_cols = safe_score_cols(df, SCORE_COLS)
    group_cols = [
        "project_code", "Tool_level4",
        "Intervention_level3", "intervention_category_level2"
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    agg        = {c: "mean" for c in score_cols}

    result = df.groupby(group_cols, as_index=False).agg(agg)
    result = round_scores(result, score_cols)
    result["summary_level"] = "L2 - Intervention Category"
    print(f"  L2: {len(df)} rows → {len(result)} aggregated rows")
    return result


# ── Sheet: L1 — Activity (most granular, sample top/bottom per project) ──
def aggregate_l1(df: pd.DataFrame) -> pd.DataFrame:
    """
    For L1 we keep one row per unique activity per project.
    This captures every distinct activity without repeating the same
    activity across multiple response rows.
    """
    score_cols = safe_score_cols(df, SCORE_COLS)
    group_cols = [
        "project_code", "Tool_level4",
        "Intervention_level3", "intervention_category_level2",
        "Activity_level1"
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    agg        = {c: "mean" for c in score_cols}

    result = df.groupby(group_cols, as_index=False).agg(agg)
    result = round_scores(result, score_cols)
    result["summary_level"] = "L1 - Activity"
    print(f"  L1: {len(df)} rows → {len(result)} aggregated rows")
    return result


# ── Project-level overview (cross-sheet summary) ──────────────────────
def build_project_overview(l4: pd.DataFrame) -> pd.DataFrame:
    """
    One row per project_code summarising average scores across all tools.
    This powers questions like 'how does project X score overall?'
    """
    score_cols = safe_score_cols(l4, SCORE_COLS + IMPACT_COLS + EFFECTIVENESS_COLS)
    agg        = {c: "mean" for c in score_cols}

    result = l4.groupby("project_code", as_index=False).agg(agg)
    result = round_scores(result, score_cols)

    # Add a plain-English summary column for each project
    def summarise_row(row):
        parts = [f"Project: {row['project_code']}"]
        if "Relevance_overall" in row:
            parts.append(f"Relevance: {row['Relevance_overall']}")
        if "Efficiency_overall" in row:
            parts.append(f"Efficiency: {row['Efficiency_overall']}")
        if "Sustainability_overall" in row:
            parts.append(f"Sustainability: {row['Sustainability_overall']}")
        if "Impact_WA" in row:
            parts.append(f"Impact: {row['Impact_WA']}")
        if "Effectiveness_WA" in row:
            parts.append(f"Effectiveness: {row['Effectiveness_WA']}")
        return " | ".join(str(p) for p in parts)

    result["plain_summary"] = result.apply(summarise_row, axis=1)
    result["summary_level"] = "Project Overview"
    print(f"  Overview: {len(result)} projects summarised")
    return result


# ── Main ──────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌  {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE} …")
    xl = pd.ExcelFile(INPUT_FILE)

    sheet_map = {s: xl.parse(s) for s in xl.sheet_names}

    results = {}

    print("Aggregating …")

    if "L4 - Tool" in sheet_map:
        results["L4_Tool"]        = aggregate_l4(sheet_map["L4 - Tool"])
        results["Project_Overview"] = build_project_overview(results["L4_Tool"])

    if "L3 - Intervention" in sheet_map:
        results["L3_Intervention"] = aggregate_l3(sheet_map["L3 - Intervention"])

    if "L2 - Intervention Category" in sheet_map:
        results["L2_Category"]     = aggregate_l2(sheet_map["L2 - Intervention Category"])

    if "L1 - Activity" in sheet_map:
        results["L1_Activity"]     = aggregate_l1(sheet_map["L1 - Activity"])

    # Write all sheets to output file
    print(f"\nWriting {OUTPUT_FILE} …")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✅  {sheet_name}: {len(df)} rows written")

    total_rows = sum(len(df) for df in results.values())
    print(f"\n✅  Done — {total_rows} total rows across {len(results)} sheets")
    print(f"   (was ~14,620 rows, now {total_rows} — "
          f"{round((1 - total_rows/14620)*100)}% reduction)")
    print(f"\nNext steps:")
    print(f"  1. Delete the original: del data\\scoring_dashboard.xlsx")
    print(f"  2. Rebuild index:       python -c \"from index_manager import "
          f"incremental_update; incremental_update(force_rebuild=True)\"")


if __name__ == "__main__":
    main()
