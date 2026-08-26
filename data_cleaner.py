# data_cleaner.py
# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive data cleaning module with adaptive strategies for different
# data types, missing value handling, and detailed cleaning reports.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime


class DataCleaningReport:
    """Tracks all cleaning operations for transparency and audit."""
    def __init__(self):
        self.operations: List[Dict[str, Any]] = []
        self.original_shape: Tuple[int, int] = (0, 0)
        self.final_shape: Tuple[int, int] = (0, 0)
        self.columns_analyzed: int = 0
        self.rows_removed: int = 0
        self.cells_filled: int = 0
        self.column_reports: Dict[str, Dict[str, Any]] = {}

    def add_operation(self, operation_type: str, details: Dict[str, Any]):
        """Log a cleaning operation."""
        self.operations.append({
            "type": operation_type,
            "timestamp": datetime.now().isoformat(),
            **details
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "rows_removed": self.rows_removed,
            "cells_filled": self.cells_filled,
            "columns_analyzed": self.columns_analyzed,
            "operations": self.operations,
            "column_reports": self.column_reports,
            "summary": f"Cleaned {self.original_shape[0]} rows × {self.original_shape[1]} columns "
                      f"→ {self.final_shape[0]} rows × {self.final_shape[1]} columns. "
                      f"Removed {self.rows_removed} duplicates, filled {self.cells_filled} missing values.",
        }


class DataCleaner:
    """Main data cleaning orchestrator with adaptive strategies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.report = DataCleaningReport()
        self.report.original_shape = df.shape
        self.report.columns_analyzed = len(df.columns)

    # ─────────────────────────────────────────────────────────────────────────
    # COLUMN STANDARDIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def standardize_column_names(self) -> "DataCleaner":
        """
        Normalize column names:
        - Convert to lowercase
        - Replace spaces with underscores
        - Remove special characters
        - Remove leading/trailing underscores
        """
        original_names = self.df.columns.tolist()
        
        new_names = []
        for col in self.df.columns:
            # Convert to lowercase
            col = str(col).lower()
            # Replace spaces and hyphens with underscores
            col = re.sub(r'[\s\-]+', '_', col)
            # Remove special characters except underscore
            col = re.sub(r'[^a-z0-9_]', '', col)
            # Remove leading/trailing underscores
            col = col.strip('_')
            # Replace multiple underscores with single
            col = re.sub(r'_+', '_', col)
            new_names.append(col)

        self.df.columns = new_names
        
        changed = sum(1 for o, n in zip(original_names, new_names) if o != n)
        if changed > 0:
            self.report.add_operation(
                "standardize_columns",
                {"columns_changed": changed, "mapping": dict(zip(original_names, new_names))}
            )
        
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # DUPLICATE REMOVAL
    # ─────────────────────────────────────────────────────────────────────────

    def remove_duplicates(self, subset: List[str] = None, keep: str = 'first') -> "DataCleaner":
        """
        Remove duplicate rows.
        
        Args:
            subset: List of column names to consider. None = all columns.
            keep: 'first', 'last', or False (remove all duplicates).
        """
        initial_rows = len(self.df)
        
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        rows_removed = initial_rows - len(self.df)
        self.report.rows_removed = rows_removed
        
        if rows_removed > 0:
            self.report.add_operation(
                "remove_duplicates",
                {
                    "rows_removed": rows_removed,
                    "subset": subset or "all_columns",
                    "keep": keep
                }
            )
        
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # DATA TYPE DETECTION & INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def filter_rows(self, column: str, operator: str, value) -> "DataCleaner":
        """
        KEEPS rows matching the condition (drops the rest). operator: equals,
        not_equals, greater_than, less_than, greater_than_equal,
        less_than_equal, contains, is_null, not_null.
        "remove rows where X is 0" -> operator="not_equals", value=0.
        """
        before = len(self.df)
        s = self.df[column]
        num = pd.to_numeric(s, errors='coerce')
        try:
            v = float(value)
            use_num = True
        except (TypeError, ValueError):
            v = value
            use_num = False

        ops = {
            'equals': lambda: (num == v) if use_num else (s.astype(str) == str(v)),
            'not_equals': lambda: (num != v) if use_num else (s.astype(str) != str(v)),
            'greater_than': lambda: num > v,
            'less_than': lambda: num < v,
            'greater_than_equal': lambda: num >= v,
            'less_than_equal': lambda: num <= v,
            'contains': lambda: s.astype(str).str.contains(str(value), case=False, na=False),
            'is_null': lambda: s.isnull(),
            'not_null': lambda: s.notnull(),
        }
        self.df = self.df[ops[operator]()].reset_index(drop=True)
        self.report.add_operation(
            "filter_rows",
            {"column": column, "operator": operator, "value": value,
             "rows_before": before, "rows_after": len(self.df), "rows_removed": before - len(self.df)}
        )
        return self

    def run_steps(self, steps: List[Dict[str, Any]]) -> "DataCleaner":
        """
        Executes an ORDERED list of steps, one at a time, e.g.:
          [{"op":"standardize_columns"},
           {"op":"filter_rows","column":"rating","operator":"not_equals","value":0},
           {"op":"handle_missing_values","strategy":"smart","columns":["rating"]},
           {"op":"remove_duplicates","subset":["id"]}]
        Unlike run_full_pipeline() (fixed order), steps run in the order given.
        """
        dispatch = {
            "standardize_columns": lambda s: self.standardize_column_names(),
            "remove_duplicates": lambda s: self.remove_duplicates(subset=s.get("subset"), keep=s.get("keep", "first")),
            "filter_rows": lambda s: self.filter_rows(s["column"], s.get("operator", "not_equals"), s.get("value")),
            "handle_missing_values": lambda s: self.handle_missing_values(
                s.get("strategy", "smart"), s.get("columns"), s.get("custom_value")
            ),
            "normalize_text": lambda s: self.normalize_text_columns(),
            "handle_outliers": lambda s: self.handle_outliers(s.get("method", "cap")),
            "infer_types": lambda s: self.infer_and_convert_types(),
            "remove_empty_rows": lambda s: self.remove_rows_with_all_nulls(),
        }
        for step in steps:
            op = step.get("op")
            fn = dispatch.get(op)
            if fn is None:
                self.report.add_operation("unknown_step", {"op": op, "error": "skipped"})
                continue
            try:
                fn(step)
            except Exception as e:
                self.report.add_operation(op, {"error": str(e)})
        return self

    def detect_column_types(self) -> Dict[str, str]:
        """
        Intelligently detect column data types:
        - numeric: integer, float, percent
        - datetime: dates, timestamps
        - categorical: limited unique values
        - text: free-form strings
        """
        type_map = {}
        
        for col in self.df.columns:
            # Skip all-NaN columns
            if self.df[col].isna().all():
                type_map[col] = "empty"
                continue
            
            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                type_map[col] = "empty"
                continue
            
            # Try numeric
            try:
                pd.to_numeric(non_null, errors='raise')
                type_map[col] = "numeric"
                continue
            except (ValueError, TypeError):
                pass
            
            # Try datetime
            try:
                pd.to_datetime(non_null.astype(str), errors='raise')
                type_map[col] = "datetime"
                continue
            except (ValueError, TypeError):
                pass
            
            # Check if categorical (limited unique values)
            unique_count = non_null.nunique()
            total_count = len(non_null)
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            
            if unique_count <= 10 or uniqueness_ratio < 0.05:
                type_map[col] = "categorical"
            else:
                type_map[col] = "text"
        
        return type_map

    # ─────────────────────────────────────────────────────────────────────────
    # MISSING VALUE HANDLING
    # ─────────────────────────────────────────────────────────────────────────

    def handle_missing_values(self, strategy: str = "smart", columns: List[str] = None, custom_value: Any = None) -> "DataCleaner":
        """
        Fill missing values using adaptive strategies.
        
        Args:
            strategy: 'smart' (auto-detect), 'mean', 'median', 'mode', 'forward_fill', 'drop', 'custom'
            columns: restrict to these columns only (default: all columns)
            custom_value: exact literal to write when strategy='custom' 
        """
        column_types = self.detect_column_types()
        cells_filled = 0
        col_reports = {}
        
        for col in (columns or self.df.columns):
            # Treat both true nulls and empty/whitespace-only text as missing.
            # CSV/Excel ingestion often turns blanks into NaN, but this also
            # protects direct dataframe callers from silently leaving "" cells.
            missing_mask = self.df[col].isna()
            if self.df[col].dtype == "object":
                missing_mask = missing_mask | self.df[col].astype("string").str.strip().eq("").fillna(False)
            missing_count = int(missing_mask.sum())
            if missing_count == 0:
                continue
            
            col_type = column_types.get(col, "unknown")
            col_report = {
                "column": col,
                "type": col_type,
                "missing_count": int(missing_count),
                "strategy_used": None,
                "fill_value": None,
            }
            
            # Explicit literal fills must also work when every value is missing.
            if strategy == "custom":
                if custom_value is None:
                    raise ValueError("custom_value must be provided when strategy='custom'")
                self.df.loc[missing_mask, col] = custom_value
                col_report["strategy_used"] = "custom"
                col_report["fill_value"] = str(custom_value)
                col_report["filled_count"] = missing_count
                cells_filled += missing_count
                col_reports[col] = col_report
                continue

            # All missing — statistical strategies have nothing to infer from.
            if missing_count == len(self.df):
                col_report["strategy_used"] = "dropped"
                col_reports[col] = col_report
                continue

            try:
                if strategy == "smart":
                    if col_type == "numeric":
                        fill_value = self.df[col].median()
                        self.df[col] = self.df[col].fillna(fill_value)
                        col_report["strategy_used"] = "median"
                        col_report["fill_value"] = float(fill_value)
                    
                    elif col_type == "categorical":
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                        self.df[col] = self.df[col].fillna(fill_value)
                        col_report["strategy_used"] = "mode"
                        col_report["fill_value"] = str(fill_value)
                    
                    elif col_type == "datetime":
                        fill_value = self.df[col].median()
                        self.df[col] = self.df[col].fillna(fill_value)
                        col_report["strategy_used"] = "median_datetime"
                        col_report["fill_value"] = str(fill_value)
                    
                    else:  # text or empty
                        fill_value = "Unknown"
                        self.df[col] = self.df[col].fillna(fill_value)
                        col_report["strategy_used"] = "placeholder"
                        col_report["fill_value"] = fill_value
                
                elif strategy == "mean":
                    if col_type == "numeric":
                        fill_value = self.df[col].mean()
                        self.df[col] = self.df[col].fillna(fill_value)
                        col_report["strategy_used"] = "mean"
                        col_report["fill_value"] = float(fill_value)
                    else:
                        self.df[col] = self.df[col].fillna("Unknown")
                        col_report["strategy_used"] = "placeholder"
                        col_report["fill_value"] = "Unknown"
                
                elif strategy == "median":
                    if col_type == "numeric":
                        fill_value = self.df[col].median()
                        self.df[col] = self.df[col].fillna(fill_value)
                        col_report["strategy_used"] = "median"
                        col_report["fill_value"] = float(fill_value)
                    else:
                        self.df[col] = self.df[col].fillna("Unknown")
                        col_report["strategy_used"] = "placeholder"
                        col_report["fill_value"] = "Unknown"
                
                elif strategy == "mode":
                    if not self.df[col].mode().empty:
                        fill_value = self.df[col].mode()[0]
                        self.df[col] = self.df[col].fillna(fill_value)
                        col_report["strategy_used"] = "mode"
                        col_report["fill_value"] = str(fill_value)
                    else:
                        self.df[col] = self.df[col].fillna("Unknown")
                        col_report["strategy_used"] = "placeholder"
                        col_report["fill_value"] = "Unknown"
                
                elif strategy == "forward_fill":
                    self.df[col] = self.df[col].ffill()
                    self.df[col] = self.df[col].fillna("Unknown")
                    col_report["strategy_used"] = "forward_fill"
                
                elif strategy == "drop":
                    self.df = self.df.dropna(subset=[col])
                    col_report["strategy_used"] = "dropped_rows"
                
                cells_filled += int(missing_count)
                col_reports[col] = col_report
            
            except Exception as e:
                col_report["strategy_used"] = "error"
                col_report["error"] = str(e)
                col_reports[col] = col_report
        
        self.report.cells_filled = cells_filled
        self.report.column_reports = col_reports
        
        if cells_filled > 0:
            self.report.add_operation(
                "handle_missing_values",
                {
                    "strategy": strategy,
                    "cells_filled": cells_filled,
                    "column_details": col_reports
                }
            )
        
        self.df.reset_index(drop=True, inplace=True)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT NORMALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def normalize_text_columns(self) -> "DataCleaner":
        """
        Clean text columns:
        - Strip leading/trailing whitespace
        - Normalize whitespace (multiple spaces → single space)
        - Remove special Unicode characters (except common punctuation)
        - Consistent casing (optional)
        """
        text_cols = self.df.select_dtypes(include=['object']).columns
        normalized_count = 0
        
        for col in text_cols:
            if self.df[col].isna().all():
                continue
            
            original = self.df[col].copy()

            # Pandas may back text columns with pyarrow strings, whose regex
            # engine is stricter than Python's and can choke on Unicode escape
            # classes. Keep the cleanup purely in Python so the behavior stays
            # stable across pandas backends.
            def _clean_text(value):
                if not isinstance(value, str):
                    return value
                value = value.strip()
                value = re.sub(r"\s+", " ", value)
                value = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", value)
                return value

            self.df[col] = self.df[col].astype("object").map(_clean_text)
            
            # Check if column changed
            if not (self.df[col] == original).all():
                normalized_count += 1
        
        if normalized_count > 0:
            self.report.add_operation(
                "normalize_text",
                {"columns_normalized": normalized_count}
            )
        
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # OUTLIER DETECTION & HANDLING
    # ─────────────────────────────────────────────────────────────────────────

    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Returns a boolean Series marking outliers as True.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)

    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Returns a boolean Series marking outliers as True.
        """
        from scipy import stats
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        return pd.Series(np.abs(stats.zscore(self.df[column])) > threshold, index=self.df.index)

    def handle_outliers(self, method: str = "cap", multiplier: float = 1.5) -> "DataCleaner":
        """
        Handle outliers in numeric columns.
        
        Args:
            method: 'cap' (set to bounds), 'remove' (drop rows), 'mark' (flag column)
            multiplier: IQR multiplier for outlier threshold
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for col in numeric_cols:
            outliers = self.detect_outliers_iqr(col, multiplier)
            outlier_count = outliers.sum()
            
            if outlier_count == 0:
                continue
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            outlier_report[col] = {
                "outlier_count": int(outlier_count),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
            
            if method == "cap":
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                outlier_report[col]["action"] = "capped"
            
            elif method == "remove":
                before = len(self.df)
                self.df = self.df[~outliers]
                removed = before - len(self.df)
                self.report.rows_removed += removed
                outlier_report[col]["action"] = f"removed_{removed}_rows"
            
            elif method == "mark":
                self.df[f"{col}_is_outlier"] = outliers
                outlier_report[col]["action"] = "marked"
        
        if outlier_report:
            self.report.add_operation(
                "handle_outliers",
                {
                    "method": method,
                    "columns_affected": len(outlier_report),
                    "details": outlier_report
                }
            )
        
        self.df.reset_index(drop=True, inplace=True)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # DATA TYPE CONVERSION
    # ─────────────────────────────────────────────────────────────────────────

    def infer_and_convert_types(self) -> "DataCleaner":
        """
        Convert columns to appropriate data types based on content.
        """
        column_types = self.detect_column_types()
        conversions = {}
        
        for col, dtype in column_types.items():
            try:
                if dtype == "numeric":
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    conversions[col] = "numeric"
                
                elif dtype == "datetime":
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    conversions[col] = "datetime"
                
                elif dtype == "categorical":
                    self.df[col] = self.df[col].astype('category')
                    conversions[col] = "categorical"
                
                else:
                    self.df[col] = self.df[col].astype(str)
                    conversions[col] = "string"
            
            except Exception as e:
                conversions[col] = f"error: {str(e)}"
        
        if conversions:
            self.report.add_operation(
                "infer_types",
                {"conversions": conversions}
            )
        
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION & CONSTRAINTS
    # ─────────────────────────────────────────────────────────────────────────

    def remove_rows_with_all_nulls(self) -> "DataCleaner":
        """Remove rows where all values are null."""
        before = len(self.df)
        self.df = self.df.dropna(how='all')
        removed = before - len(self.df)
        
        if removed > 0:
            self.report.rows_removed += removed
            self.report.add_operation(
                "remove_all_null_rows",
                {"rows_removed": removed}
            )
        
        return self

    def remove_rows_with_null_threshold(self, threshold: float = 0.5) -> "DataCleaner":
        """
        Remove rows where more than threshold% of values are null.
        
        Args:
            threshold: Fraction of nulls to trigger removal (0.0-1.0)
        """
        before = len(self.df)
        null_fractions = self.df.isna().sum(axis=1) / len(self.df.columns)
        self.df = self.df[null_fractions <= threshold]
        removed = before - len(self.df)
        
        if removed > 0:
            self.report.rows_removed += removed
            self.report.add_operation(
                "remove_high_null_rows",
                {
                    "threshold": threshold,
                    "rows_removed": removed
                }
            )
        
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLETE CLEANING PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        standardize_cols: bool = True,
        remove_dups: bool = True,
        remove_empty_rows: bool = True,
        handle_nulls: bool = True,
        null_strategy: str = "smart",
        normalize_text: bool = True,
        infer_types: bool = True,
        handle_outliers_flag: bool = False,
        outlier_method: str = "cap",
    ) -> "DataCleaner":
        """
        Run the complete cleaning pipeline with configurable steps.
        """
        if standardize_cols:
            self.standardize_column_names()
        
        if remove_dups:
            self.remove_duplicates()
        
        if remove_empty_rows:
            self.remove_rows_with_all_nulls()
        
        if handle_nulls:
            self.handle_missing_values(strategy=null_strategy)
        
        if normalize_text:
            self.normalize_text_columns()
        
        if infer_types:
            self.infer_and_convert_types()
        
        if handle_outliers_flag:
            self.handle_outliers(method=outlier_method)
        
        self.report.final_shape = self.df.shape
        
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # GETTERS
    # ─────────────────────────────────────────────────────────────────────────

    def get_cleaned_dataframe(self) -> pd.DataFrame:
        """Return the cleaned dataframe."""
        return self.df

    def get_report(self) -> DataCleaningReport:
        """Return the cleaning report."""
        return self.report

    def get_report_dict(self) -> Dict[str, Any]:
        """Return the cleaning report as a dictionary."""
        return self.report.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION FOR ROUTE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataframe(
    df: pd.DataFrame,
    config: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    High-level function to clean a dataframe with configuration.
    
    Args:
        df: Input dataframe
        config: Dictionary with cleaning parameters:
            - standardize_cols: bool (default True)
            - remove_duplicates: bool (default True)
            - remove_empty_rows: bool (default True)
            - handle_missing_values: bool (default True)
            - null_strategy: str ('smart', 'mean', 'median', 'mode', 'forward_fill', 'drop')
            - normalize_text: bool (default True)
            - infer_types: bool (default True)
            - handle_outliers: bool (default False)
            - outlier_method: str ('cap', 'remove', 'mark')
    
    Returns:
        Tuple of (cleaned_dataframe, report_dict)
    """
    if config is None:
        config = {}
    
    cleaner = DataCleaner(df)
    if config.get('steps'):
        cleaner.run_steps(config['steps'])
        return cleaner.get_cleaned_dataframe(), cleaner.get_report_dict()
    cleaner.run_full_pipeline(
        standardize_cols=config.get('standardize_cols', True),
        remove_dups=config.get('remove_duplicates', True),
        remove_empty_rows=config.get('remove_empty_rows', True),
        handle_nulls=config.get('handle_missing_values', True),
        null_strategy=config.get('null_strategy', 'smart'),
        normalize_text=config.get('normalize_text', True),
        infer_types=config.get('infer_types', True),
        handle_outliers_flag=config.get('handle_outliers', False),
        outlier_method=config.get('outlier_method', 'cap'),
    )
    
    return cleaner.get_cleaned_dataframe(), cleaner.get_report_dict()
