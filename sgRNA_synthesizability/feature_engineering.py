from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TOKEN_PATTERN = re.compile(r"/[^/]+/|\+[ACGUT]|[mr][ACGUT]|[ACGUT]|\*")
BASES = ("A", "C", "G", "U", "T")
RNA_TOKENS = tuple(f"r{base}" for base in "ACGU")
METHYL_TOKENS = tuple(f"m{base}" for base in "ACGU")
LNA_TOKENS = tuple(f"l{base}" for base in "ACGUT")
CHEMISTRY_TOKENS = RNA_TOKENS + METHYL_TOKENS + LNA_TOKENS
WORK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = WORK_DIR / "data" / "cleaned_data.csv"
DEFAULT_OUTPUT_DIR = WORK_DIR / "analysis" / "features"
TARGET_COLUMN = "Pass/Fail"
SEQUENCE_COLUMN = "Sequence"
ID_COLUMN = "Ref ID"
POSITIVE_LABEL = "Fail"
NEGATIVE_LABEL = "Pass"


@dataclass(frozen=True)
class ParsedSequence:
    tokens: list[str]
    bases: str
    unknown_chars: str


def parse_decorated_sequence(sequence: str) -> ParsedSequence:
    sequence = "" if pd.isna(sequence) else str(sequence)
    tokens: list[str] = []
    bases: list[str] = []
    consumed = [False] * len(sequence)

    for match in TOKEN_PATTERN.finditer(sequence):
        raw_token = match.group(0)
        token = f"l{raw_token[1]}" if raw_token.startswith("+") else raw_token
        tokens.append(token)
        for index in range(match.start(), match.end()):
            consumed[index] = True

        if token in BASES:
            bases.append("U" if token == "T" else token)
        elif len(token) == 2 and token[0] == "l" and token[1] in "ACGUT":
            bases.append("U" if token[1] == "T" else token[1])
        elif len(token) == 2 and token[0] in {"m", "r"} and token[1] in "ACGUT":
            bases.append("U" if token[1] == "T" else token[1])

    unknown_chars = "".join(
        char for char, was_consumed in zip(sequence, consumed) if not was_consumed and not char.isspace()
    )
    return ParsedSequence(tokens=tokens, bases="".join(bases), unknown_chars=unknown_chars)


def longest_run(sequence: str, alphabet: set[str] | None = None) -> int:
    best = 0
    current = 0
    previous = None
    for char in sequence:
        if alphabet is not None and char not in alphabet:
            current = 0
            previous = None
            continue
        if char == previous:
            current += 1
        else:
            current = 1
            previous = char
        best = max(best, current)
    return best


def longest_flag_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def max_gc_window_fraction(sequence: str, window_size: int = 10) -> float:
    if not sequence:
        return 0.0
    if len(sequence) <= window_size:
        return gc_fraction(sequence)

    best = 0.0
    for start in range(0, len(sequence) - window_size + 1):
        window = sequence[start : start + window_size]
        best = max(best, gc_fraction(window))
    return best


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(base in {"G", "C"} for base in sequence) / len(sequence)


def fraction(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def window_values(values: list[int]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    length = len(values)
    first_end = max(1, length // 3)
    second_end = max(first_end + 1, (2 * length) // 3) if length > 1 else length
    windows = [values[:first_end], values[first_end:second_end], values[second_end:]]
    return tuple(sum(window) / len(window) if window else 0.0 for window in windows)


def position_summary(positions: list[int], length: int) -> dict[str, float | int]:
    if not positions or length <= 0:
        return {
            "first_position": -1,
            "last_position": -1,
            "mean_position": -1.0,
            "position_std": 0.0,
            "first_fraction": -1.0,
            "last_fraction": -1.0,
            "mean_fraction": -1.0,
            "span_fraction": 0.0,
        }

    denominator = max(1, length - 1)
    mean_position = sum(positions) / len(positions)
    variance = sum((position - mean_position) ** 2 for position in positions) / len(positions)
    return {
        "first_position": min(positions),
        "last_position": max(positions),
        "mean_position": mean_position,
        "position_std": variance**0.5,
        "first_fraction": min(positions) / denominator,
        "last_fraction": max(positions) / denominator,
        "mean_fraction": mean_position / denominator,
        "span_fraction": (max(positions) - min(positions)) / denominator,
    }


def window_counts(positions: list[int], length: int) -> tuple[int, int, int, float, float, float]:
    if length <= 0:
        return 0, 0, 0, 0.0, 0.0, 0.0

    first_end = max(1, length // 3)
    second_end = max(first_end + 1, (2 * length) // 3) if length > 1 else length
    windows = [(0, first_end), (first_end, second_end), (second_end, length)]
    counts = [sum(start <= position < end for position in positions) for start, end in windows]
    lengths = [max(0, end - start) for start, end in windows]
    densities = [count / window_length if window_length else 0.0 for count, window_length in zip(counts, lengths)]
    return counts[0], counts[1], counts[2], densities[0], densities[1], densities[2]


def add_position_features(
    features: dict[str, int | float],
    prefix: str,
    positions: list[int],
    length: int,
) -> None:
    for name, value in position_summary(positions, length).items():
        features[f"{prefix}_{name}"] = value

    count_5p, count_middle, count_3p, density_5p, density_middle, density_3p = window_counts(positions, length)
    features[f"{prefix}_count_5p"] = count_5p
    features[f"{prefix}_count_middle"] = count_middle
    features[f"{prefix}_count_3p"] = count_3p
    features[f"{prefix}_density_5p"] = density_5p
    features[f"{prefix}_density_middle"] = density_middle
    features[f"{prefix}_density_3p"] = density_3p


def sequence_features(sequence: str) -> dict[str, int | float]:
    parsed = parse_decorated_sequence(sequence)
    tokens = parsed.tokens
    bases = parsed.bases
    token_counts = Counter(tokens)
    base_counts = Counter(bases)
    token_length = len(tokens)
    base_length = len(bases)
    modification_flags = [int(token.startswith(("m", "l", "/")) or token == "*") for token in tokens]
    mod_5p, mod_middle, mod_3p = window_values(modification_flags)
    star_positions = [index for index, token in enumerate(tokens) if token == "*"]
    slash_mod_positions = [index for index, token in enumerate(tokens) if token.startswith("/")]
    methyl_positions = [index for index, token in enumerate(tokens) if token.startswith("m")]
    lna_positions = [index for index, token in enumerate(tokens) if token.startswith("l")]
    rna_positions = [index for index, token in enumerate(tokens) if token.startswith("r")]
    modified_positions = [index for index, is_modified in enumerate(modification_flags) if is_modified]
    star_flags = [token == "*" for token in tokens]
    slash_mod_flags = [token.startswith("/") for token in tokens]
    methyl_flags = [token.startswith("m") for token in tokens]
    lna_flags = [token.startswith("l") for token in tokens]
    rna_flags = [token.startswith("r") for token in tokens]
    modified_flags = [bool(is_modified) for is_modified in modification_flags]

    features: dict[str, int | float] = {
        "decorated_length": len(str(sequence)),
        "token_length": token_length,
        "base_length": base_length,
        "unknown_char_count": len(parsed.unknown_chars),
        "unknown_char_fraction": fraction(len(parsed.unknown_chars), len(str(sequence))),
        "slash_mod_count": sum(token.startswith("/") for token in tokens),
        "has_slash_mod": int(any(token.startswith("/") for token in tokens)),
        "star_count": token_counts["*"],
        "star_density": fraction(token_counts["*"], token_length),
        "terminal_star_count": int(bool(tokens[:1] and tokens[0] == "*")) + int(bool(tokens[-1:] and tokens[-1] == "*")),
        "mod_density_5p": mod_5p,
        "mod_density_middle": mod_middle,
        "mod_density_3p": mod_3p,
        "gc_fraction": gc_fraction(bases),
        "au_fraction": fraction(base_counts["A"] + base_counts["U"], base_length),
        "purine_fraction": fraction(base_counts["A"] + base_counts["G"], base_length),
        "pyrimidine_fraction": fraction(base_counts["C"] + base_counts["U"] + base_counts["T"], base_length),
        "terminal_gc_count": int(bool(bases[:1] and bases[0] in {"G", "C"})) + int(bool(bases[-1:] and bases[-1] in {"G", "C"})),
        "longest_homopolymer_run": longest_run(bases),
        "longest_gc_run": longest_run(bases, {"G", "C"}),
        "max_gc_window_10": max_gc_window_fraction(bases, window_size=10),
        "repeated_dinucleotide_count": sum(
            1 for index in range(max(0, base_length - 3)) if bases[index : index + 2] == bases[index + 2 : index + 4]
        ),
        "longest_star_run": longest_flag_run(star_flags),
        "longest_slash_mod_run": longest_flag_run(slash_mod_flags),
        "longest_methyl_run": longest_flag_run(methyl_flags),
        "longest_lna_run": longest_flag_run(lna_flags),
        "longest_rna_run": longest_flag_run(rna_flags),
        "longest_modified_token_run": longest_flag_run(modified_flags),
    }

    for base in BASES:
        normalized_base = "U" if base == "T" else base
        count = base_counts[normalized_base]
        features[f"base_{base}_count"] = count
        features[f"base_{base}_fraction"] = fraction(count, base_length)

    for token in CHEMISTRY_TOKENS:
        features[f"token_{token}_count"] = token_counts[token]
        features[f"token_{token}_fraction"] = fraction(token_counts[token], token_length)
        features[f"longest_{token}_run"] = longest_flag_run([observed_token == token for observed_token in tokens])

    add_position_features(features, "star", star_positions, token_length)
    add_position_features(features, "slash_mod", slash_mod_positions, token_length)
    add_position_features(features, "methyl", methyl_positions, token_length)
    add_position_features(features, "lna", lna_positions, token_length)
    add_position_features(features, "rna", rna_positions, token_length)
    add_position_features(features, "modified_token", modified_positions, token_length)

    return features


def build_feature_table(data: pd.DataFrame, sequence_column: str = "Sequence") -> pd.DataFrame:
    features = [sequence_features(sequence) for sequence in data[sequence_column].fillna("")]
    return pd.DataFrame(features, index=data.index)


def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string")


def clean_modeling_data(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    sequence_column: str = SEQUENCE_COLUMN,
) -> pd.DataFrame:
    cleaned = data.copy().replace("", pd.NA)
    cleaned = cleaned.dropna(subset=[target_column, sequence_column])
    cleaned = cleaned[cleaned[target_column].isin([NEGATIVE_LABEL, POSITIVE_LABEL])]
    return cleaned.reset_index(drop=False).rename(columns={"index": "source_row_index"})


def encode_context_features(
    data: pd.DataFrame,
    exclude_columns: set[str] | None = None,
) -> pd.DataFrame:
    exclude_columns = exclude_columns or set()
    context = data.drop(columns=[column for column in exclude_columns if column in data.columns], errors="ignore")

    if context.empty:
        return pd.DataFrame(index=data.index)

    numeric_features = pd.DataFrame(index=data.index)
    categorical_features = pd.DataFrame(index=data.index)

    for column in context.columns:
        numeric_values = pd.to_numeric(context[column], errors="coerce")
        numeric_fraction = numeric_values.notna().mean()
        if numeric_fraction >= 0.95:
            fill_value = numeric_values.median()
            if pd.isna(fill_value):
                fill_value = 0.0
            numeric_features[f"context_{column}"] = numeric_values.fillna(fill_value).astype(float)
        else:
            categorical_features[column] = context[column].fillna("Unknown").astype("string")

    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features, prefix="context", dtype=int)

    return pd.concat([numeric_features, categorical_features], axis=1)


def prepare_modeling_data(
    data: pd.DataFrame,
    include_context_features: bool = False,
    target_column: str = TARGET_COLUMN,
    sequence_column: str = SEQUENCE_COLUMN,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    modeling_data = clean_modeling_data(data, target_column=target_column, sequence_column=sequence_column)
    sequence_feature_table = build_feature_table(modeling_data, sequence_column=sequence_column)

    if include_context_features:
        context_features = encode_context_features(
            modeling_data,
            exclude_columns={"source_row_index", id_column, target_column, sequence_column},
        )
        features = pd.concat([sequence_feature_table, context_features], axis=1)
    else:
        features = sequence_feature_table

    target = modeling_data[target_column].eq(POSITIVE_LABEL).astype(int).rename("target")
    groups = modeling_data[sequence_column].rename("group_sequence")
    return features, target, groups, modeling_data


def write_feature_outputs(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    modeling_data: pd.DataFrame,
    output_dir: Path,
    input_path: Path,
    include_context_features: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_output = features.copy()
    feature_output.insert(0, "id", modeling_data[ID_COLUMN] if ID_COLUMN in modeling_data.columns else pd.NA)
    feature_output.to_csv(output_dir / "feature_matrix.csv", index=False)
    pd.DataFrame(
        {
            "source_row_index": modeling_data["source_row_index"],
            ID_COLUMN: modeling_data[ID_COLUMN] if ID_COLUMN in modeling_data.columns else pd.NA,
            "target": target,
            "label": modeling_data[TARGET_COLUMN],
        }
    ).to_csv(output_dir / "target_vector.csv", index=False)
    pd.DataFrame(
        {
            "source_row_index": modeling_data["source_row_index"],
            "group_sequence": groups,
        }
    ).to_csv(output_dir / "groups.csv", index=False)

    metadata = {
        "input_path": str(input_path),
        "rows": int(len(modeling_data)),
        "features": int(features.shape[1]),
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "positive_rows": int(target.sum()),
        "negative_rows": int((target == 0).sum()),
        "include_context_features": include_context_features,
        "id_column": ID_COLUMN,
        "feature_matrix_id_column": "id",
        "model_feature_columns_exclude_id": True,
        "unknown_char_rows": int((features.get("unknown_char_count", pd.Series(dtype=int)) > 0).sum()),
        "feature_columns": list(features.columns),
    }
    (output_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sgRNA feature matrix and target vector for modeling.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for feature outputs.")
    parser.add_argument(
        "--include-context-features",
        action="store_true",
        help="Encode non-sequence, non-target columns as additional model features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_raw_data(args.input)
    features, target, groups, modeling_data = prepare_modeling_data(
        data,
        include_context_features=args.include_context_features,
    )
    write_feature_outputs(
        features,
        target,
        groups,
        modeling_data,
        args.output_dir,
        args.input,
        args.include_context_features,
    )
    print(f"rows={len(modeling_data)} features={features.shape[1]} positives={int(target.sum())}")
    print(f"saved_outputs={args.output_dir}")


if __name__ == "__main__":
    main()
