from __future__ import annotations

from pathlib import Path

import pandas as pd


WORK_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = WORK_DIR / "data" / "all.cleaned_data.csv"
OUTPUT_PATH = WORK_DIR / "data" / "cleaned_data.mainpeak.csv"
SOURCE_COLUMNS = ["Ref ID", "UHPLC % MainPeak", "Sequence"]
OUTPUT_COLUMNS = ["Ref ID", "MainPeak", "Sequence"]


def main() -> None:
    data = pd.read_csv(INPUT_PATH, dtype="string")
    missing_columns = [column for column in SOURCE_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    output = data[SOURCE_COLUMNS].rename(columns={"UHPLC % MainPeak": "MainPeak"})
    output = output[OUTPUT_COLUMNS]
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"rows={len(output)} columns={list(output.columns)}")
    print(f"missing_mainpeak={int(output['MainPeak'].isna().sum())} missing_sequence={int(output['Sequence'].isna().sum())}")
    print(f"saved_output={OUTPUT_PATH.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()