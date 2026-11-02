"""Assemble the combined report's tables from the two tracks' generated tables.

  python reports/build_tables.py      # after pretrained/scripts/report_tables.py, classical/report_tables.py
                                      # and common/compare_tracks.py

The combined report covers the six submitted models (3 pretrained + 3 classical). This script only
selects and concatenates rows that the track scripts generated from results files; it computes nothing.
Output: reports/tables/*.tex.
"""
from __future__ import annotations

from pathlib import Path

TEAM = Path(__file__).resolve().parent.parent
PRE, CLA = TEAM / "pretrained" / "reports" / "tables", TEAM / "classical" / "reports" / "tables"
OUT = Path(__file__).resolve().parent / "tables"


def rows(path: Path, keep=None) -> list[str]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [ln for ln in lines if keep is None or any(k in ln for k in keep)]


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote reports/tables/{name} ({len(lines)} lines)")


def main():
    # Q1.3: the two classical headline models + the LightGBM head shared by the pretrained blends' LM models
    write("q13_hpsearch.tex", rows(CLA / "q13_hpsearch.tex", [r"\texttt{tfidf\_lr}", r"\texttt{stylo\_gbm}"])
          + rows(PRE / "q13_hpsearch.tex", [r"\texttt{lm\_gpt2}"]))
    # Q3: lm_gpt2 (pretrained) and both classical models
    write("q32_semisup.tex", rows(PRE / "q32_semisup.tex", [r"\texttt{lm\_gpt2}"]) + ["\\midrule"]
          + rows(CLA / "q32_semisup.tex"))
    # Appendix: the log is shared, either track's copy is the same table
    write("appendix_submissions.tex", rows(PRE / "appendix_submissions.tex"))


if __name__ == "__main__":
    main()
