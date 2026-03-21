"""Export translation_v2 JSONL results to a human-review CSV template."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_COLUMNS = [
    "id",
    "lang",
    "source_type",
    "edit_type",
    "instruction_en",
    "original_translation",
    "translated_text_v2",
    "backtranslate_to_en",
    "qa_flag",
    "qa_score",
    "qa_notes",
    "human_translation_ok",
    "human_semantic_preservation_ok",
    "failure_type",
    "notes",
]

SOURCE_FIELDS = {
    "id",
    "lang",
    "source_type",
    "edit_type",
    "instruction_en",
    "original_translation",
    "translated_text_v2",
    "backtranslate_to_en",
    "qa_flag",
    "qa_score",
    "qa_notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a translation_v2 JSONL file into a review CSV."
    )
    parser.add_argument("--input", required=True, help="Path to translation_v2 JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output review CSV file.")
    return parser.parse_args()


def export_review_csv(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {input_path}: {exc}"
                ) from exc

            row = {column: "" for column in CSV_COLUMNS}
            for field in SOURCE_FIELDS:
                value = record.get(field, "")
                row[field] = "" if value is None else value

            writer.writerow(row)
            rows_written += 1

    return rows_written


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        rows_written = export_review_csv(input_path=input_path, output_path=output_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
