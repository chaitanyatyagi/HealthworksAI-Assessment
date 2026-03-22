import pandas as pd
import requests
import json
import re

# -------------------------------
# CONFIG
# -------------------------------
INTENT_MODEL = "qwen2.5-coder:1.5b"  # Fast, precise for JSON
REASONING_MODEL = "deepseek-r1:latest" # Deep thinking for complex benefits

# -------------------------------
# 1. THE "MASTER" PROMPT (Intent Extraction)
# -------------------------------
def get_intent_and_filters(query, columns):
    """
    Analyzes the user query to extract semantic meaning and filters.
    """
    prompt = f"""
You are an AI Benefit Specialist. Analyze the user's query and the available data schema.

SCHEMA:
{columns}

EXTRACTION RULES:
1. 'bid_id': Look for patterns like HXXXX-XXX-XXX.
2. 'entities': Extract the main benefit being asked about (e.g., "Ambulance", "Hearing").
3. 'intent': Is the user asking for "cost", "limitations", "prior authorization", or a "general summary"?
4. 'synonyms': Provide 2-3 synonyms for the benefit (e.g., "Hearing" -> ["Audiology", "Hearing Aids", "Ear"]).

RETURN ONLY VALID JSON:
{{
  "bid_id": "extracted_id_or_null",
  "primary_benefit": "extracted_benefit",
  "search_keywords": ["synonym1", "synonym2"],
  "specific_questions": ["cost", "limitations"]
}}

USER QUERY: {query}
JSON:"""

    try:
        response = requests.post("http://localhost:11434/api/generate", 
            json={"model": INTENT_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        ).json()
        return json.loads(re.sub(r"```json|```", "", response.get("response", "")))
    except:
        return None

# -------------------------------
# 2. THE "INTELLIGENT" RETRIEVER
# -------------------------------
def retrieve_data(df, intent):
    """
    Performs a multi-layered search: Strict on Plan ID, Fuzzy on Benefits.
    """
    if not intent:
        return df.head(5)

    res = df.copy()
    
    # Layer 1: Filter by Plan ID (if provided)
    if intent.get("bid_id"):
        res = res[res['bid_id'].astype(str).str.contains(intent['bid_id'], na=False)]

    # Layer 2: Semantic Keyword Search
    # We search the keywords across BOTH 'benefit_name' and 'service_type'
    keywords = [intent.get("primary_benefit")] + intent.get("search_keywords", [])
    keywords = [k for k in keywords if k] # Remove nulls
    
    if keywords:
        pattern = '|'.join(keywords)
        mask = (
            res['benefit_name'].str.contains(pattern, case=False, na=False) |
            res['service_type'].str.contains(pattern, case=False, na=False)
        )
        res = res[mask]

    # Layer 3: Quality Control
    # If the search results are too broad, we keep the top 15 most relevant rows
    return res.head(15)

# -------------------------------
# 3. THE "REASONING" FORMATTER
# -------------------------------
def generate_final_response(query, context_df):
    """
    Uses DeepSeek-R1 to analyze the raw data and write a human-readable answer.
    """
    if context_df.empty:
        return "I'm sorry, I couldn't find specific details for that benefit in the selected plan. Would you like me to check a different service?"

    # Convert DF to a clean string format for the LLM
    data_context = context_df.to_csv(index=False)
    
    prompt = f"""
<role>
You are a Senior Health Insurance Advisor. Your goal is to provide accurate, easy-to-understand benefit information.
</role>

<context>
User Question: {query}
Raw Data from Database:
{data_context}
</context>

<instructions>
1. If the data shows multiple options (e.g., Ground vs Air Ambulance), list both clearly.
2. Explicitly state if "Prior Authorization" is required.
3. If "Cost Sharing" is $0 or "No", phrase it as "This service is covered at no cost to you."
4. Use a professional tone. If limitations exist (e.g., 'once per year'), highlight them.
</instructions>

Answer:"""

    try:
        response = requests.post("http://localhost:11434/api/generate", 
            json={"model": REASONING_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.5}}
        ).json()
        return response.get("response", "")
    except Exception as e:
        return f"Data found, but synthesis failed: {str(e)}"
