from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WORK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WORK_DIR / "data"
INPUT_FILES = (
    DATA_DIR / "all.data1.csv",
    DATA_DIR / "all.data2.csv",
    DATA_DIR / "all.data3.csv",
)
OUTPUT_PATH = DATA_DIR / "all.cleaned_data.csv"
SUMMARY_PATH = DATA_DIR / "all.cleaned_data_summary.json"
VALID_LABELS = {"Pass", "Fail", "Needs Review"}
LABEL_PRIORITY = {"Fail": 0, "Needs Review": 1, "Pass": 2}


def sanitize_text(value: object) -> object:
    if pd.isna(value):
        return value
    if not isinstance(value, str):
        return value
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


def normalize_column_name(column: str) -> str:
    return " ".join(str(column).replace("\xa0", " ").split())


def read_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", encoding="cp1252")
    frame = frame.rename(columns={column: normalize_column_name(column) for column in frame.columns})
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed")]
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    for column in frame.select_dtypes(include="string").columns:
        frame[column] = frame[column].map(sanitize_text)
    frame["source_file"] = path.name
    return frame


def row_completeness(frame: pd.DataFrame) -> pd.Series:
    return frame.notna().sum(axis=1)


def clean_combined_data() -> tuple[pd.DataFrame, dict[str, object]]:
    input_frames = [read_input(path) for path in INPUT_FILES]
    raw_combined = pd.concat(input_frames, ignore_index=True, sort=False)
    raw_rows = len(raw_combined)
    raw_nonempty_ref_rows = int(raw_combined["Ref ID"].notna().sum())

    combined = raw_combined.dropna(how="all").copy()
    combined = combined.dropna(subset=["Ref ID", "Pass/Fail", "Sequence"]).copy()
    combined = combined[combined["Pass/Fail"].isin(VALID_LABELS)].copy()
    combined = combined.drop_duplicates().copy()
    combined["label_priority"] = combined["Pass/Fail"].map(LABEL_PRIORITY).fillna(99).astype(int)
    combined["row_completeness"] = row_completeness(combined)
    combined["source_priority"] = combined["source_file"].map(
        {path.name: index for index, path in enumerate(INPUT_FILES)}
    ).fillna(len(INPUT_FILES)).astype(int)

    duplicate_ref_rows = int(combined["Ref ID"].duplicated(keep=False).sum())
    label_conflict_ids = int((combined.groupby("Ref ID")["Pass/Fail"].nunique(dropna=True) > 1).sum())
    combined = combined.sort_values(
        ["Ref ID", "label_priority", "row_completeness", "source_priority"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    cleaned = combined.drop_duplicates(subset=["Ref ID"], keep="first").copy()
    cleaned = cleaned.drop(columns=["label_priority", "row_completeness", "source_priority"])

    front_columns = [column for column in ["Ref ID", "Pass/Fail", "Sequence", "source_file"] if column in cleaned.columns]
    remaining_columns = [column for column in cleaned.columns if column not in front_columns]
    cleaned = cleaned[front_columns + remaining_columns]
    cleaned = cleaned.sort_values("Ref ID", kind="mergesort").reset_index(drop=True)
    for column in cleaned.select_dtypes(include="string").columns:
        cleaned[column] = cleaned[column].map(sanitize_text)

    summary = {
        "input_files": [str(path.relative_to(WORK_DIR)) for path in INPUT_FILES],
        "output_file": str(OUTPUT_PATH.relative_to(WORK_DIR)),
        "raw_rows": raw_rows,
        "raw_nonempty_ref_rows": raw_nonempty_ref_rows,
        "rows_after_required_field_filter": int(len(combined)),
        "duplicate_ref_rows_before_deduplication": duplicate_ref_rows,
        "label_conflict_ref_ids_before_deduplication": label_conflict_ids,
        "cleaned_rows": int(len(cleaned)),
        "unique_ref_ids": int(cleaned["Ref ID"].nunique()),
        "label_counts": {str(label): int(count) for label, count in cleaned["Pass/Fail"].value_counts(dropna=False).items()},
        "source_counts": {str(source): int(count) for source, count in cleaned["source_file"].value_counts(dropna=False).items()},
        "deduplication_rule": "Sort each Ref ID by Pass/Fail priority Fail, Needs Review, Pass; then keep the most complete row; then prefer earlier input file order.",
        "cell_text_sanitization": "Embedded newline and carriage-return characters are replaced with single spaces in string cells.",
    }
    return cleaned, summary


def main() -> None:
    cleaned, summary = clean_combined_data()
    cleaned.to_csv(OUTPUT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"raw_rows={summary['raw_rows']} raw_nonempty_ref_rows={summary['raw_nonempty_ref_rows']}")
    print(f"cleaned_rows={summary['cleaned_rows']} unique_ref_ids={summary['unique_ref_ids']}")
    print(f"label_counts={summary['label_counts']}")
    print(f"saved_output={OUTPUT_PATH.relative_to(WORK_DIR)}")
    print(f"saved_summary={SUMMARY_PATH.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()