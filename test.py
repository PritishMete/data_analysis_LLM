import pandas as pd
from pathlib import Path

def analyze_file(file_path):
    file_path = Path(file_path)

    # Detect file type
    file_type = file_path.suffix.lower()

    print(f"Detected file type: {file_type}")

    # Load file based on type
    if file_type == ".csv":
        df = pd.read_csv(file_path) 

    elif file_type in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif file_type == ".json":
        df = pd.read_json(file_path)

    elif file_type == ".parquet":
        df = pd.read_parquet(file_path)

    else:
        print("Unsupported file type")
        return

    # Show dataset information
    print("\n===== DATA INFO =====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


# column names and preview
    print("\nColumn Names:")
    for col in df.columns:
        print(f"- {col}")

# top 5 rows
    print("\nPreview:")
    print(df.head())

# sample data
    print("\n===== SAMPLE DATA =====")
    print(df.sample())

# info about dataset
    print("\n===== DATASET INFO =====")
    print(df.info())

# descriptive statistics
    print("\n===== DESCRIPTIVE STATISTICS =====")   
    print(df.describe())

# duplicate rows
    print(df.duplicated().sum())

# missing values
    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

# not a number values
    print("\n===== NOT A NUMBER (NaN) VALUES =====")
    print(df.isna().sum())
# unique values
    print("\n===== UNIQUE VALUES =====")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")


# Example
analyze_file("apps.csv")