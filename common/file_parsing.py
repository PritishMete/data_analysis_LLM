# common/file_parsing.py
# ─────────────────────────────────────────────────────────────────────────────
# Single shared implementation of "raw upload bytes -> pandas DataFrame",
# extracted so the Dataset Registry doesn't duplicate the private
# `_read_file_to_df()` that already lives in agentic_cleaning_routes.py.
#
# NOTE (integration, not required to adopt this feature): agentic_cleaning_
# routes.py currently has its own copy of this exact logic as a module-private
# `_read_file_to_df()`. It still works fine as-is — nothing here requires
# touching that file. If/when you want to de-duplicate it, the one-line
# change is swapping its body for a call to `read_file_to_dataframe()` below;
# that's a follow-up you can make on your own schedule, not something this
# feature depends on.
# ─────────────────────────────────────────────────────────────────────────────

import io

import pandas as pd


def read_file_to_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    """Parses CSV/TSV/XLSX/XLS/XLSM bytes into a DataFrame. Falls back to CSV
    parsing for unrecognized extensions, matching the existing behavior in
    agentic_cleaning_routes.py so results are identical either way."""
    name = (filename or "").lower()
    if name.endswith(".csv") or name.endswith(".tsv"):
        sep = "\t" if name.endswith(".tsv") else ","
        return pd.read_csv(io.BytesIO(raw_bytes), sep=sep)
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(io.BytesIO(raw_bytes))
    return pd.read_csv(io.BytesIO(raw_bytes))
