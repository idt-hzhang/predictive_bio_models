from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).resolve().parents[1]
ALL_CLEANED_DATA_PATH = WORK_DIR / "data" / "all.cleaned_data.csv"
CLEANED_DATA_PATH = WORK_DIR / "data" / "cleaned_data.csv"
NEED_REVIEW_PATH = WORK_DIR / "data" / "cleaned_data.need_review.csv"
TABLE_PATH = WORK_DIR / "data" / "pass_fail_need_review_table.csv"


def need_review_label(length: pd.Series, mainpeak: pd.Series) -> pd.Series:
    thresholds = pd.Series(np.nan, index=length.index, dtype=float)
    thresholds[length < 56] = 80.0
    thresholds[(length >= 56) & (length < 90)] = 75.0
    thresholds[(length >= 90) & (length < 120)] = 70.0
    thresholds[length >= 120] = 50.0

    labels = pd.Series(pd.NA, index=length.index, dtype="string")
    valid = mainpeak.notna() & thresholds.notna()
    labels.loc[valid & (mainpeak >= thresholds)] = "Pass"
    labels.loc[valid & (mainpeak < thresholds)] = "Needs Review"
    return labels


def write_need_review_dataset() -> pd.DataFrame:
    data = pd.read_csv(ALL_CLEANED_DATA_PATH, dtype="string")
    length = pd.to_numeric(data["Length"], errors="coerce")
    mainpeak = pd.to_numeric(data["UHPLC % MainPeak"], errors="coerce")
    need_review = data[["Ref ID", "Sequence"]].copy()
    need_review.insert(1, "Need_Review", need_review_label(length, mainpeak))
    need_review.to_csv(NEED_REVIEW_PATH, index=False)
    return need_review


def write_pass_fail_need_review_table(need_review: pd.DataFrame) -> pd.DataFrame:
    pass_fail = pd.read_csv(CLEANED_DATA_PATH, dtype="string", usecols=["Ref ID", "Pass/Fail"])
    comparison = pass_fail.merge(need_review[["Ref ID", "Need_Review"]], on="Ref ID", how="left", validate="one_to_one")
    comparison = comparison[
        comparison["Pass/Fail"].isin(["Pass", "Fail"])
        & comparison["Need_Review"].isin(["Pass", "Needs Review"])
    ].copy()

    table = (
        comparison.groupby(["Pass/Fail", "Need_Review"], dropna=False)
        .size()
        .reindex(pd.MultiIndex.from_product([["Pass", "Fail"], ["Pass", "Needs Review"]], names=["Pass/Fail", "Need_Review"]), fill_value=0)
        .reset_index(name="count")
    )
    total = int(table["count"].sum())
    row_totals = table.groupby("Pass/Fail")["count"].transform("sum")
    column_totals = table.groupby("Need_Review")["count"].transform("sum")
    table["row_proportion"] = table["count"] / row_totals
    table["column_proportion"] = table["count"] / column_totals
    table["overall_proportion"] = table["count"] / total
    table["row_total"] = row_totals
    table["column_total"] = column_totals
    table["overall_total"] = total
    table.to_csv(TABLE_PATH, index=False)
    return table


def main() -> None:
    need_review = write_need_review_dataset()
    table = write_pass_fail_need_review_table(need_review)
    print(f"need_review_rows={len(need_review)}")
    print(need_review["Need_Review"].value_counts(dropna=False).to_string())
    print("\nPass/Fail x Need_Review table:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved={NEED_REVIEW_PATH.relative_to(WORK_DIR)}")
    print(f"saved={TABLE_PATH.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()