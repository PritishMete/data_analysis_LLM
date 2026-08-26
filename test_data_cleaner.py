# test_data_cleaner.py
# ─────────────────────────────────────────────────────────────────────────────
# Test and usage examples for the data cleaning module.
# Run this locally to validate the cleaning logic before deploying to Render.
#
# Usage:
#   python test_data_cleaner.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from data_cleaner import DataCleaner, clean_dataframe

# =============================================================================
# TEST 1: Basic Sales Data Cleaning
# =============================================================================

def test_sales_data():
    """Test cleaning of a messy sales dataset."""
    print("\n" + "="*80)
    print("TEST 1: SALES DATA CLEANING")
    print("="*80)
    
    # Create messy sales data
    df = pd.DataFrame({
        'Order ID': [1, 2, 3, 3, 4, 5, None, 6],  # Has duplicate & null
        'Customer Name': ['  John Smith  ', 'john smith', 'JOHN SMITH', 'JOHN SMITH', 
                         'Jane Doe', 'jane doe', 'Bob Brown', None],
        'Amount': [1000.50, 1000.50, 1000.50, np.nan, 2500.0, np.nan, 5000.0, 3000.0],
        'Status': ['Completed', None, 'Completed', 'Completed', None, 'Pending', 'Completed', 'Pending'],
        'Date': ['2024-01-15', '2024-01-15', '2024-01-15', '01/15/2024', 
                '2024-01-16', '01/16/2024', '2024-01-17', None],
    })
    
    print("\nORIGINAL DATA:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Clean the data
    config = {
        "standardize_cols": True,
        "remove_duplicates": True,
        "remove_empty_rows": True,
        "handle_missing_values": True,
        "null_strategy": "smart",
        "normalize_text": True,
        "infer_types": True,
        "handle_outliers": True,
        "outlier_method": "cap",
    }
    
    cleaned_df, report = clean_dataframe(df, config)
    
    print("\n\nCLEANED DATA:")
    print(cleaned_df)
    print(f"\nShape: {cleaned_df.shape}")
    print(f"Missing values:\n{cleaned_df.isnull().sum()}")
    print(f"Duplicates: {cleaned_df.duplicated().sum()}")
    
    print("\n\nCLEANING REPORT:")
    print(f"Summary: {report['summary']}")
    print(f"Rows removed: {report['rows_removed']}")
    print(f"Cells filled: {report['cells_filled']}")
    
    print("\nColumn Reports:")
    for col, col_report in report['column_reports'].items():
        print(f"\n  {col}:")
        print(f"    - Type: {col_report['type']}")
        print(f"    - Missing: {col_report['missing_count']}")
        print(f"    - Strategy: {col_report['strategy_used']}")
        print(f"    - Fill value: {col_report['fill_value']}")


# =============================================================================
# TEST 2: Customer Database with Deduplication
# =============================================================================

def test_customer_deduplication():
    """Test deduplication of customer records."""
    print("\n" + "="*80)
    print("TEST 2: CUSTOMER DATABASE DEDUPLICATION")
    print("="*80)
    
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'Name': ['Alice Johnson', 'Alice Johnson', 'Bob Smith', 'ALICE JOHNSON', 'Carol White', 'Bob Smith'],
        'Email': ['alice@test.com', 'alice@test.com', 'bob@test.com', 'alice@test.com', 'carol@test.com', 'bob@test.com'],
        'Status': ['Active', 'Active', 'Active', 'Active', None, None],
    })
    
    print("\nORIGINAL DATA:")
    print(df)
    print(f"Duplicates: {df.duplicated().sum()}")
    
    config = {
        "standardize_cols": True,
        "remove_duplicates": True,
        "handle_missing_values": True,
        "null_strategy": "mode",
        "normalize_text": True,
        "infer_types": False,
    }
    
    cleaned_df, report = clean_dataframe(df, config)
    
    print("\n\nDEDUPLICATED DATA:")
    print(cleaned_df)
    print(f"Rows removed: {report['rows_removed']}")


# =============================================================================
# TEST 3: Numeric Data with Outliers
# =============================================================================

def test_outlier_handling():
    """Test detection and handling of outliers."""
    print("\n" + "="*80)
    print("TEST 3: OUTLIER DETECTION & HANDLING")
    print("="*80)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'Product': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        'Price': [100, 105, 98, 102, 9999, 101, 99, 103],  # 9999 is outlier
        'Quantity': [10, 12, 8, 11, 150, 9, 13, 11],  # 150 is outlier
    })
    
    print("\nORIGINAL DATA (with outliers):")
    print(df)
    
    config = {
        "standardize_cols": True,
        "handle_outliers": True,
        "outlier_method": "cap",
    }
    
    cleaned_df, report = clean_dataframe(df, config)
    
    print("\n\nDATA WITH CAPPED OUTLIERS:")
    print(cleaned_df)
    print(f"\nOutlier handling: {report['operations']}")


# =============================================================================
# TEST 4: Time Series with Forward Fill
# =============================================================================

def test_time_series_forward_fill():
    """Test forward filling for time series data."""
    print("\n" + "="*80)
    print("TEST 4: TIME SERIES FORWARD FILL")
    print("="*80)
    
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=8),
        'Sensor_A': [22.5, None, 23.1, 23.2, None, None, 24.0, 24.1],
        'Sensor_B': [50, 51, None, 52, 53, None, 54, 55],
    })
    
    print("\nORIGINAL DATA (with gaps):")
    print(df)
    print(f"Missing values:\n{df.isnull().sum()}")
    
    config = {
        "handle_missing_values": True,
        "null_strategy": "forward_fill",
        "infer_types": True,
    }
    
    cleaned_df, report = clean_dataframe(df, config)
    
    print("\n\nFORWARD FILLED DATA:")
    print(cleaned_df)
    print(f"Cells filled: {report['cells_filled']}")


# =============================================================================
# TEST 5: Mixed Types Detection
# =============================================================================

def test_type_inference():
    """Test automatic type detection and conversion."""
    print("\n" + "="*80)
    print("TEST 5: AUTOMATIC TYPE INFERENCE")
    print("="*80)
    
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Carol'],
        'Age': ['25', '30', '28'],  # Numeric as strings
        'Salary': ['50000.00', '60000.00', '55000.00'],  # Numeric as strings
        'JoinDate': ['2023-01-15', '2023-06-20', '2023-03-10'],  # Dates as strings
        'Status': ['Active', 'Inactive', 'Active'],  # Categorical
    })
    
    print("\nORIGINAL DATA (all strings):")
    print(df)
    print(f"Data types:\n{df.dtypes}")
    
    config = {
        "infer_types": True,
    }
    
    cleaned_df, report = clean_dataframe(df, config)
    
    print("\n\nINFERRED TYPES:")
    print(f"Data types:\n{cleaned_df.dtypes}")
    print(f"\nType conversions: {report['operations']}")


# =============================================================================
# TEST 6: Column Name Standardization
# =============================================================================

def test_name_standardization():
    """Test column name cleaning and standardization."""
    print("\n" + "="*80)
    print("TEST 6: COLUMN NAME STANDARDIZATION")
    print("="*80)
    
    df = pd.DataFrame({
        'Customer Name': [1, 2, 3],
        'Sale $ Amount': [100, 200, 300],
        'Report-Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Status (Active/Inactive)': ['A', 'I', 'A'],
    })
    
    print(f"\nORIGINAL COLUMNS: {list(df.columns)}")
    
    config = {
        "standardize_cols": True,
    }
    
    cleaned_df, report = clean_dataframe(df, config)
    
    print(f"CLEANED COLUMNS: {list(cleaned_df.columns)}")


# =============================================================================
# TEST 7: API-style Test (Simulating Backend Call)
# =============================================================================

def test_api_style():
    """Test simulating an API call with JSON configuration."""
    print("\n" + "="*80)
    print("TEST 7: API-STYLE CLEANING (JSON Config)")
    print("="*80)
    
    # Simulate file data
    df = pd.DataFrame({
        'ID': [1, 1, 2, 3, np.nan],
        'Name': ['  Alice  ', 'alice', 'Bob', None, 'Carol'],
        'Value': [100.5, 100.5, None, 200.0, 150.0],
    })
    
    print("\nORIGINAL DATA:")
    print(df)
    
    # Simulate JSON config from API
    config = {
        "standardize_cols": True,
        "remove_duplicates": True,
        "remove_empty_rows": True,
        "handle_missing_values": True,
        "null_strategy": "smart",
        "normalize_text": True,
        "infer_types": True,
        "handle_outliers": False,
        "outlier_method": "cap",
        "output_sheet_name": "Cleaned_Data"
    }
    
    cleaned_df, report = clean_dataframe(df, config)
    
    print("\n\nCLEANED DATA:")
    print(cleaned_df)
    
    # Simulate export format
    export_format = {
        "sheet_name": config["output_sheet_name"],
        "columns": list(cleaned_df.columns),
        "rows": cleaned_df.fillna("").to_dict(orient="records"),
        "row_count": len(cleaned_df),
    }
    
    print("\n\nEXPORT FORMAT (ready for Excel):")
    print(f"Sheet: {export_format['sheet_name']}")
    print(f"Rows: {export_format['row_count']}")
    print(f"Columns: {export_format['columns']}")


# =============================================================================
# TEST 8: Comparison of Different Null Strategies
# =============================================================================

def test_null_strategies():
    """Compare different null-filling strategies."""
    print("\n" + "="*80)
    print("TEST 8: NULL STRATEGY COMPARISON")
    print("="*80)
    
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Price': [100.0, None, None, 200.0, 150.0],
        'Status': ['Active', None, 'Inactive', None, 'Active'],
    })
    
    print("\nORIGINAL DATA:")
    print(df)
    
    strategies = ['smart', 'mean', 'median', 'mode', 'drop']
    
    for strategy in strategies:
        config = {
            "handle_missing_values": True,
            "null_strategy": strategy,
        }
        
        try:
            cleaned_df, report = clean_dataframe(df.copy(), config)
            print(f"\n\n{strategy.upper()}:")
            print(cleaned_df)
            print(f"Cells filled: {report['cells_filled']}")
            print(f"Rows after: {len(cleaned_df)}")
        except Exception as e:
            print(f"\n{strategy.upper()}: ERROR - {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " DATA CLEANER TEST SUITE ".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    # Run all tests
    test_sales_data()
    test_customer_deduplication()
    test_outlier_handling()
    test_time_series_forward_fill()
    test_type_inference()
    test_name_standardization()
    test_api_style()
    test_null_strategies()
    
    print("\n" + "█"*80)
    print("█" + " ALL TESTS COMPLETE ".center(78) + "█")
    print("█"*80 + "\n")


# =============================================================================
# BONUS: REST API Test Examples
# =============================================================================

"""
To test the backend API, use these curl commands:

1. BASIC CLEANING:
curl -X POST https://data-analysis-oajs.onrender.com/clean_data \
  -F "file=@data.csv" \
  -F "config={\"standardize_cols\":true,\"remove_duplicates\":true}"

2. FULL CLEANING:
curl -X POST https://data-analysis-oajs.onrender.com/clean_data \
  -F "file=@data.xlsx" \
  -F "config={\"standardize_cols\":true,\"remove_duplicates\":true,\"handle_missing_values\":true,\"null_strategy\":\"smart\",\"infer_types\":true}"

3. WITH OUTLIER HANDLING:
curl -X POST https://data-analysis-oajs.onrender.com/clean_data \
  -F "file=@sales.csv" \
  -F "config={\"handle_outliers\":true,\"outlier_method\":\"cap\"}"


Python client example:

import requests
import json

config = {
    "standardize_cols": True,
    "remove_duplicates": True,
    "handle_missing_values": True,
    "null_strategy": "smart",
    "infer_types": True,
}

files = {'file': open('data.csv', 'rb')}
data = {'config': json.dumps(config)}

response = requests.post(
    'https://data-analysis-oajs.onrender.com/clean_data',
    files=files,
    data=data
)

result = response.json()
print(f"Success: {result['success']}")
print(f"Summary: {result['summary']}")
print(f"Rows: {result['before']['rows']} → {result['after']['rows']}")
"""
