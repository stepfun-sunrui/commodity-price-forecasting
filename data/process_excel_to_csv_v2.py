"""\
Rebuild indicator CSVs from raw Excel with correct de-dup logic.

Why v2:
- The old `process_excel_to_csv.py` writes per sheet sequentially, so duplicated 指标编码 will be
  silently overwritten by later sheets (e.g. ID00102866 ends up taking '新增指标' instead of '预测标的').

What this script does:
- Parses each sheet by locating the rows labeled '指标名称' / '指标编码' and the first date row.
- For each 指标编码, collects all candidate columns across sheets.
- Chooses ONE "best" candidate by default using:
  1) earliest first non-null date
  2) then more non-null points
  3) then later last date
  4) then earlier sheet order
- Writes per-code CSV with 2 header rows + (date,value) rows.
- Writes a report `generation_report.csv` describing duplicates and the chosen source.

Default output goes to `indicator_data_fixed/` to avoid overwriting existing data.

Example:
  python process_excel_to_csv_v2.py --only-code ID00102866
  python process_excel_to_csv_v2.py --output indicator_data  # overwrite existing
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    sheet: str
    sheet_index: int
    col_index: int
    series: pd.Series  # index: datetime, values: float

    @property
    def start_date(self) -> pd.Timestamp:
        return self.series.index.min()

    @property
    def end_date(self) -> pd.Timestamp:
        return self.series.index.max()

    @property
    def points(self) -> int:
        return int(self.series.shape[0])


def _find_row_index(df: pd.DataFrame, label: str) -> int | None:
    col0 = df.iloc[:, 0].astype(str)
    matches = col0[col0 == label]
    if matches.empty:
        return None
    return int(matches.index[0])


def _find_first_date_row(df: pd.DataFrame) -> int | None:
    # Excel timestamp cells are often already datetime-like; parse explicitly to reduce warnings
    col0 = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    valid = col0[col0.notna()]
    if valid.empty:
        return None
    return int(valid.index[0])


def _normalize_str(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _extract_candidates_from_sheet(df: pd.DataFrame, sheet: str, sheet_index: int) -> list[Candidate]:
    name_row = _find_row_index(df, "指标名称")
    code_row = _find_row_index(df, "指标编码")
    data_start = _find_first_date_row(df)

    if name_row is None or data_start is None:
        return []

    dates = pd.to_datetime(df.iloc[data_start:, 0], errors="coerce")

    candidates: list[Candidate] = []
    for col in range(1, df.shape[1]):
        name = _normalize_str(df.iloc[name_row, col])
        if not name:
            continue

        code = _normalize_str(df.iloc[code_row, col]) if code_row is not None else ""
        if not code:
            # Skip no-code columns in v2 (they are hard to join/dedupe correctly)
            continue

        values = pd.to_numeric(df.iloc[data_start:, col], errors="coerce")
        tmp = pd.DataFrame({"date": dates, "value": values}).dropna(subset=["date", "value"])
        if tmp.empty:
            continue

        tmp = tmp.sort_values("date")
        tmp = tmp.drop_duplicates(subset=["date"], keep="last")
        series = pd.Series(tmp["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(tmp["date"]))

        candidates.append(
            Candidate(
                code=code,
                name=name,
                sheet=sheet,
                sheet_index=sheet_index,
                col_index=col,
                series=series,
            )
        )

    return candidates


def _candidate_sort_key(c: Candidate):
    # earlier start is better; then more points; then later end; then earlier sheet order
    end_ns = int(c.end_date.value) if isinstance(c.end_date, pd.Timestamp) else -1
    return (c.start_date, -c.points, -end_ns, c.sheet_index)


def _mismatch_stats(primary: Candidate, other: Candidate) -> tuple[int, int]:
    """Returns (overlap_points, different_points) on overlapping dates."""
    left = primary.series
    right = other.series
    common = left.index.intersection(right.index)
    if len(common) == 0:
        return 0, 0
    lv = left.loc[common].to_numpy()
    rv = right.loc[common].to_numpy()
    diff = np.abs(lv - rv) > 1e-9
    return int(len(common)), int(diff.sum())


def write_indicator_csv(path: Path, code: str, name: str, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # match existing style: newest first, datetime with time part
    s = series.sort_index(ascending=False)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["指标名称", name])
        writer.writerow(["指标编码", code])
        for dt, val in s.items():
            writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S"), float(val)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel",
        default="raw/焦煤特征因子指标-数据v3.0(2).xlsx",
        help="Path to the raw Excel file",
    )
    parser.add_argument(
        "--output",
        default="processed",
        help="Output directory (default: processed)",
    )
    parser.add_argument(
        "--only-code",
        default="",
        help="If set, only rebuild this 指标编码 (e.g. ID00102866)",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    out_dir = Path(args.output)

    xls = pd.ExcelFile(excel_path)
    all_candidates: dict[str, list[Candidate]] = {}

    for sheet_index, sheet_name in enumerate(xls.sheet_names):
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        candidates = _extract_candidates_from_sheet(df, sheet_name, sheet_index)
        for c in candidates:
            if args.only_code and c.code != args.only_code:
                continue
            all_candidates.setdefault(c.code, []).append(c)

    if not all_candidates:
        print("No candidates found. Check the excel path / code.")
        return

    report_rows: list[dict[str, object]] = []

    for code, cands in sorted(all_candidates.items()):
        cands_sorted = sorted(cands, key=_candidate_sort_key)
        chosen = cands_sorted[0]

        # write chosen series
        out_path = out_dir / f"{code}.csv"
        write_indicator_csv(out_path, code=code, name=chosen.name, series=chosen.series)

        # report
        duplicates = len(cands_sorted)
        others_desc = " | ".join([f"{c.sheet}(col={c.col_index},start={c.start_date.date()},n={c.points})" for c in cands_sorted[1:]])
        overlap_total = 0
        diff_total = 0
        for other in cands_sorted[1:]:
            overlap, diff = _mismatch_stats(chosen, other)
            overlap_total += overlap
            diff_total += diff

        report_rows.append(
            {
                "code": code,
                "chosen_sheet": chosen.sheet,
                "chosen_col": chosen.col_index,
                "start": chosen.start_date.strftime("%Y-%m-%d"),
                "end": chosen.end_date.strftime("%Y-%m-%d"),
                "points": chosen.points,
                "duplicates": duplicates,
                "overlap_points_with_others": overlap_total,
                "diff_points_with_others": diff_total,
                "other_candidates": others_desc,
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_path = out_dir / "generation_report.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"Done. Wrote {len(report_rows)} indicator CSVs to {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

