import re
import pandas as pd
import traceback
# Import your specific LLM client here (e.g., from ai_engine import llm_generate)

# ── Security fix ──────────────────────────────────────────────────────────
# The previous version called exec(generated_code, {}, execution_env) with a
# comment claiming this was "restricted to pandas". It was not: CPython
# auto-injects the FULL `__builtins__` module into an empty globals dict at
# exec() time if the key is missing, so the generated code still had
# unrestricted access to __import__, open, eval, exec, etc. Since
# `generated_code` comes from an LLM completion (and the prompt embeds
# user-controlled column names / sample data, so prompt injection is a real
# path to influence what gets generated), this was effectively unrestricted
# code execution.
#
# Fix: (1) an explicit allow-list of harmless builtins actually needed for
# typical vectorised pandas fill expressions, so `__builtins__` is never the
# real module; (2) a pre-execution deny-list rejecting obviously dangerous
# tokens as defense in depth (not a substitute for #1 — a determined
# bypass of a string blocklist is possible, which is exactly why #1, not
# this, is the real boundary).
_SAFE_BUILTINS = {
    "len": len, "range": range, "int": int, "float": float, "str": str,
    "bool": bool, "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "True": True,
    "False": False, "None": None,
}

_DANGEROUS_PATTERNS = re.compile(
    r"\b(import|__import__|open|exec|eval|compile|globals|locals|getattr|"
    r"setattr|delattr|vars|input|__builtins__|__loader__|__class__|"
    r"__subclasses__|__bases__|os\.|sys\.|subprocess|socket|shutil)\b"
)


class UnsafeGeneratedCodeError(Exception):
    """Raised when AI-generated cleaning code fails the pre-execution safety check."""


class AgenticBacktracker:
    def __init__(self, llm_client):
        self.llm = llm_client

    def _generate_imputation_code(self, columns: list, target_column: str, sample_data: dict) -> str:
        """
        Prompts the AI to deduce the mathematical relationship and generate Pandas code.
        """
        prompt = f"""
        You are an expert AI Data Analyst. You are given a pandas DataFrame named `df`.
        
        Columns: {columns}
        Sample Data: {sample_data}
        Target Column with Missing Values: '{target_column}'
        
        Task:
        1. Deduce the mathematical relationship between '{target_column}' and the other columns based on their semantic names. 
           (e.g., if 'discount_percentage' is missing, deduce it algebraically from qty, price, and total_price).
        2. Write Python Pandas code to fill ONLY the missing (NaN) values in `df['{target_column}']`.
        3. Use vectorised Pandas operations (e.g., `df.loc[df['{target_column}'].isnull(), '{target_column}'] = ...`).
        
        Constraints:
        - Return ONLY valid, executable Python code. 
        - Do not include markdown formatting, explanations, or ```python blocks.
        - Do not overwrite existing valid data in the target column.
        """
        
        # Replace this with your actual LLM call (Gemini/Vertex AI)
        raw_response = self.llm.generate(prompt)
        
        # Clean up the response just in case the LLM outputs markdown
        clean_code = raw_response.replace("```python", "").replace("```", "").strip()
        return clean_code

    def apply_dynamic_backtrack(self, df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, bool, str]:
        """
        Executes the AI-generated code safely on the DataFrame.
        """
        if target_column not in df.columns:
            return df, False, f"Column {target_column} not found in dataset."

        columns = list(df.columns)
        # Pass a small sample so the AI understands the data scale (e.g., percentages as 0.10 vs 10)
        sample_data = df.head(3).to_dict(orient="records") 
        
        generated_code = self._generate_imputation_code(columns, target_column, sample_data)

        match = _DANGEROUS_PATTERNS.search(generated_code)
        if match:
            return df, False, (
                f"AI generated code failed the safety check (disallowed token: '{match.group(0)}'). "
                f"Code:\n{generated_code}"
            )

        # Real sandbox: __builtins__ is an explicit allow-list, never the
        # real builtins module, so even code that slips past the deny-list
        # above has no path to __import__, open, eval, etc.
        execution_env = {
            'pd': pd,
            'df': df.copy(),  # Operate on a copy to prevent partial mutations on failure
        }
        restricted_globals = {'__builtins__': _SAFE_BUILTINS}

        try:
            exec(generated_code, restricted_globals, execution_env)
            updated_df = execution_env['df']
            return updated_df, True, "Successfully backtracked and filled missing values."
        except Exception as e:
            error_trace = traceback.format_exc()
            return df, False, f"AI generated invalid code. Error: {str(e)}\nCode:\n{generated_code}"
