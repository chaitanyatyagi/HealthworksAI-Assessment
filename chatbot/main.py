import pandas as pd
import requests
import json
import re
import ast
import numpy as np

# IMPORT THE TEXT GENERATION FUNCTION
from pipeline.llm_handler import generate_raw_text 

# -------------------------------
# CONFIG
# -------------------------------
INTENT_MODEL = "qwen2.5:7b"
REASONING_MODEL = "qwen2.5:7b"

# -------------------------------
# 1. INTENT EXTRACTION — add complex_analysis category
# -------------------------------
def get_intent_and_filters(query, columns):
    prompt = f"""
You are an AI Data Router. Analyze the user's query against this CSV SCHEMA:
{columns}

Classify into ONE of these categories:
1. "column_info"      - User asks what a specific column means or contains
2. "list_values"      - User wants unique/distinct values from a specific column
3. "benefit_inquiry"  - User asks about coverage, cost, or rules for a medical service
4. "complex_analysis" - User asks for comparisons, inconsistencies, aggregations, patterns,
                        or anything that requires looking across MULTIPLE rows/columns at once.
                        Examples: "find inconsistencies", "compare X across Y", "which benefits have different Z"

RETURN ONLY VALID JSON:
{{
  "query_category": "column_info | list_values | benefit_inquiry | complex_analysis",
  "target_column": "EXACT column name if column_info or list_values, else null",
  "bid_id": "extracted plan ID or null",
  "primary_benefit": "extracted benefit name or null",
  "search_keywords": ["synonym1", "synonym2"]
}}

USER QUERY: {query}
JSON:"""

    try:
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": INTENT_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        ).json()
        raw_json = re.sub(r"```json|```", "", response.get("response", "")).strip()
        return json.loads(raw_json)
    except Exception as e:
        print(f"Intent extraction error: {e}")
        return None


# -------------------------------
# 2. DATA RETRIEVER — add complex_analysis path
# -------------------------------
def retrieve_data(df, intent):
    if not intent:
        return df.head(10), "benefit_inquiry"

    category = intent.get("query_category", "benefit_inquiry")

    # --- PATH A: COLUMN INFO ---
    if category == "column_info":
        target_col = _fuzzy_match_column(df, intent.get("target_column"))
        if target_col:
            anchor_cols = [c for c in ['benefit_name', 'service_type'] if c in df.columns]
            cols_to_show = list(dict.fromkeys(anchor_cols + [target_col]))
            result = (
                df[cols_to_show]
                .dropna(subset=[target_col])
                .drop_duplicates(subset=[target_col])
                .head(10)
            )
            return result, "column_info"
        return pd.DataFrame({"Info": [f"Column not found. Available: {', '.join(df.columns)}"]}), "column_info"

    # --- PATH B: LIST VALUES ---
    if category == "list_values":
        target_col = _fuzzy_match_column(df, intent.get("target_column"))
        if target_col:
            unique_vals = df[target_col].dropna().unique().tolist()
            return pd.DataFrame({target_col: unique_vals}), "list_values"
        return pd.DataFrame({"Info": [f"Column not found. Available: {', '.join(df.columns)}"]}), "list_values"

    # --- PATH C: COMPLEX ANALYSIS ---
    # Send a broad representative sample — no filtering, the LLM needs the full picture
    if category == "complex_analysis":
        # Cap at 80 rows to stay within context limits; prioritize diversity
        sample = df.drop_duplicates().head(80)
        return sample, "complex_analysis"

    # --- PATH D: BENEFIT INQUIRY ---
    res = df.copy()

    if intent.get("bid_id") and str(intent.get("bid_id")).lower() != "null":
        res = res[res['bid_id'].astype(str).str.contains(str(intent['bid_id']), case=False, na=False)]

    keywords = [intent.get("primary_benefit")] + intent.get("search_keywords", [])
    keywords = [str(k).strip() for k in keywords if k and str(k).strip() and str(k).lower() != "null"]

    if keywords:
        pattern = '|'.join([re.escape(k) for k in keywords])
        search_cols = [c for c in ['benefit_name', 'service_type'] if c in res.columns]
        if search_cols:
            mask = res[search_cols].apply(
                lambda col: col.astype(str).str.contains(pattern, case=False, na=False)
            ).any(axis=1)
            if not mask.any():
                mask = res.astype(str).apply(
                    lambda col: col.str.contains(pattern, case=False, na=False)
                ).any(axis=1)
            res = res[mask]

    return res.head(15), "benefit_inquiry"


# -------------------------------
# 3. RESPONSE FORMATTER — add complex_analysis prompt
# -------------------------------
def generate_final_response(query, context_df, category):
    if context_df.empty:
        return "I couldn't find any data matching your query. Try rephrasing or check the column name."

    # LIST VALUES: plain text, no LLM
    if category == "list_values":
        col = context_df.columns[0]
        values = context_df[col].tolist()
        formatted = "\n".join(f"- {v}" for v in values)
        return f"Here are all unique values for **{col}**:\n\n{formatted}"

    data_context = context_df.to_csv(index=False)

    if category == "column_info":
        instruction = "Explain what kind of information is stored in the queried column, using the provided rows as concrete examples. Be concise."

    elif category == "complex_analysis":
        instruction = """
You are analyzing a full dataset extract. Perform the analysis the user requested carefully and thoroughly.
Rules:
1. Work through the data systematically — group, compare, and aggregate as needed.
2. Clearly identify any inconsistencies, variations, or patterns.
3. Present findings in a structured, readable format (use tables or grouped lists where helpful).
4. If variations exist, explain what context (limitations, notes, plan differences) might explain them.
5. Do NOT say "I cannot access the data" — the data is provided in full above.
"""

    else:  # benefit_inquiry
        instruction = """
1. Answer the user's question directly using the raw data only.
2. If multiple options exist (e.g. Ground vs Air Ambulance), list them clearly.
3. State explicitly if Prior Authorization is required.
4. If cost sharing is $0 or None, say "covered at no cost to you."
5. Highlight any limitations (e.g. frequency limits, age restrictions).
"""

    prompt = f"""
<role>You are a Senior Health Insurance Advisor and Data Analyst.</role>

<context>
User Question: {query}

Raw Data:
{data_context}
</context>

<instructions>{instruction}</instructions>

Answer:"""

    try:
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": REASONING_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}
        ).json()
        return response.get("response", "No response from model.")
    except Exception as e:
        return f"Data found but synthesis failed: {str(e)}"

# -------------------------------
# 4. GENERIC CSV HANDLER (Safe Execution)
# -------------------------------
def handle_generic_csv(user_input, df):
    if df is None or df.empty:
        return "Error: The dataframe is empty or not loaded properly."

    columns_info = df.dtypes.astype(str).to_dict()
    sample = df.head(3).to_dict(orient='records')

    prompt = f"""
You are a Senior Python Data Engineer. You have a pandas DataFrame named 'df' already loaded in memory. DO NOT redefine or recreate 'df'.

SCHEMA & TYPES:
{columns_info}

SAMPLE DATA (First 3 rows):
{sample}

USER QUERY: "{user_input}"

TASK:
Write a Python script using the existing 'df' variable to answer the query.
Rules:
1. Store the final answer in a variable named exactly: ANSWER
2. ANSWER must be a plain Python string, list, number, or dict — NOT a DataFrame or numpy array.
3. If extracting unique values, convert to a plain Python list: .unique().tolist()
4. Do NOT redefine df, do NOT import pandas, do NOT create sample data.
5. Wrap your script strictly inside [CODE] and [/CODE] tags.

EXAMPLE:
[CODE]
clean = df['procedure_code_category'].dropna()
ANSWER = clean.unique().tolist()
[/CODE]
"""

    raw_response = generate_raw_text(prompt)

    match = re.search(r'\[CODE\](.*?)\[/CODE\]', raw_response, re.DOTALL | re.IGNORECASE)

    if match:
        code = match.group(1).strip()

        # Strip any lines that try to redefine df or import pandas
        blocked_starts = ('df =', 'df=', 'import pandas', 'import numpy', 'data =', 'data=')
        code_lines = [
            line for line in code.split('\n')
            if not any(line.strip().startswith(b) for b in blocked_starts)
        ]
        safe_code = '\n'.join(code_lines)

        exec_environment = {"df": df, "pd": pd, "np": np, "ast": ast, "ANSWER": None}

        try:
            exec(safe_code, exec_environment)
            result = exec_environment.get("ANSWER")

            if result is None:
                return "Code ran but no ANSWER was produced. Try rephrasing your query."

            # ✅ Format result as simple plain text — no LLM needed
            return format_result_as_text(user_input, result)

        except Exception as e:
            return f"Execution failed: {str(e)}\n\nScript attempted:\n```python\n{safe_code}\n```"

    return raw_response  # fallback if no code block found


def format_result_as_text(user_input, result):
    """Converts a raw Python result into a clean plain-text response. No LLM needed."""
    if isinstance(result, list):
        if len(result) == 0:
            return "No results found."
        # If it's a short list of simple values, render as bullet points
        items = "\n".join(f"- {item}" for item in result)
        return f"Here are the results for your query:\n\n{items}"
    
    elif isinstance(result, dict):
        lines = "\n".join(f"- {k}: {v}" for k, v in result.items())
        return f"Here are the results:\n\n{lines}"
    
    elif isinstance(result, (int, float)):
        return f"The result is: {result}"
    
    else:
        # Covers strings, numpy scalars, etc.
        return str(result)