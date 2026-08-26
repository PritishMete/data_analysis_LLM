"""
data_cleaning_utils.py
-----------------------
Reusable, dynamic pandas utilities for:
  1. Filling null values (mean / median / mode / custom text)
  2. Cleaning column headers (lower/upper case + trimming)
  3. Currency conversion between any two currencies

Usage examples are at the bottom of this file (under `if __name__ == "__main__"`).
"""

import pandas as pd
import numpy as np
import requests


# ---------------------------------------------------------------------------
# 1. FILL NULL VALUES
# ---------------------------------------------------------------------------
def fill_nulls(df, columns=None, method="mean", custom_value=None):
    """
    Fill null values in one or more columns dynamically.

    Parameters
    ----------
    df : pd.DataFrame
    columns : str | list | None
        Column name(s) to fill. If None, applies to all columns with nulls.
    method : str
        One of: 'mean', 'median', 'mode', 'custom'
    custom_value : any
        Used only when method='custom'. Can be text, number, etc.

    Returns
    -------
    pd.DataFrame (new copy, original untouched)
    """
    df = df.copy()

    # Normalize columns input
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    elif isinstance(columns, str):
        columns = [columns]

    method = method.lower().strip()

    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found. Skipping.")
            continue

        if method == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                print(f"⚠️ '{col}' is not numeric, cannot use mean. Skipping.")

        elif method == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                print(f"⚠️ '{col}' is not numeric, cannot use median. Skipping.")

        elif method == "mode":
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

        elif method == "custom":
            if custom_value is None:
                raise ValueError("custom_value must be provided when method='custom'")
            df[col] = df[col].fillna(custom_value)

        else:
            raise ValueError("method must be one of: 'mean', 'median', 'mode', 'custom'")

    return df


# ---------------------------------------------------------------------------
# 2. CLEAN / CHANGE HEADERS
# ---------------------------------------------------------------------------
def clean_headers(df, case="lower", trim=True, replace_spaces_with=None):
    """
    Standardize column headers dynamically.

    Parameters
    ----------
    df : pd.DataFrame
    case : str
        'lower', 'upper', or 'title' (or None to leave case unchanged)
    trim : bool
        Strip leading/trailing whitespace from headers
    replace_spaces_with : str | None
        Optional: replace spaces in headers with e.g. '_'

    Returns
    -------
    pd.DataFrame (new copy)
    """
    df = df.copy()
    new_cols = []

    for col in df.columns:
        c = str(col)

        if trim:
            c = c.strip()

        if case:
            case_l = case.lower().strip()
            if case_l == "lower":
                c = c.lower()
            elif case_l == "upper":
                c = c.upper()
            elif case_l == "title":
                c = c.title()

        if replace_spaces_with is not None:
            c = c.replace(" ", replace_spaces_with)

        new_cols.append(c)

    df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# 3. CURRENCY CONVERSION
# ---------------------------------------------------------------------------
def convert_currency(df, column, from_currency, to_currency,
                      rate=None, new_column=None, use_live_rate=False):
    """
    Convert a numeric column from one currency to another.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column containing the currency values to convert
    from_currency : str
        e.g. 'USD'
    to_currency : str
        e.g. 'INR'
    rate : float | None
        Manually supplied exchange rate (1 unit of from_currency = rate units of to_currency).
        If None and use_live_rate=False, raises an error.
    new_column : str | None
        Name of the output column. Defaults to '<column>_<to_currency>'
    use_live_rate : bool
        If True, fetches a live exchange rate from a free API (requires internet).

    Returns
    -------
    pd.DataFrame (new copy)
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if use_live_rate:
        rate = _get_live_exchange_rate(from_currency, to_currency)

    if rate is None:
        raise ValueError("Provide a 'rate' or set use_live_rate=True")

    out_col = new_column or f"{column}_{to_currency.upper()}"
    df[out_col] = df[column] * rate

    return df


def _get_live_exchange_rate(from_currency, to_currency):
    """Fetch live exchange rate using a free public API (exchangerate-api style)."""
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data["rates"][to_currency.upper()]
    except Exception as e:
        raise RuntimeError(f"Could not fetch live rate: {e}")


# ---------------------------------------------------------------------------
# DEMO / USAGE EXAMPLES
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        " Name ": ["Alice", "Bob", None, "David"],
        "Age": [25, np.nan, 30, 40],
        "Salary": [50000, 60000, np.nan, 80000],
        " City": ["NY", None, "LA", "NY"]
    })

    print("Original:\n", sample)

    filled = fill_nulls(sample, columns="Age", method="mean")
    filled = fill_nulls(filled, columns="Salary", method="median")
    filled = fill_nulls(filled, columns=" City", method="mode")
    filled = fill_nulls(filled, columns=" Name ", method="custom", custom_value="Unknown")
    print("\nAfter filling nulls:\n", filled)

    cleaned = clean_headers(filled, case="lower", trim=True, replace_spaces_with="_")
    print("\nAfter header cleanup:\n", cleaned.columns.tolist())

    converted = convert_currency(cleaned, column="salary",
                                  from_currency="USD", to_currency="INR",
                                  rate=83.5)
    print("\nAfter currency conversion:\n", converted)
