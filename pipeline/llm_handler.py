import requests
import json

# The prompt remains the same to ensure high-quality business logic extraction
PROMPT_TEMPLATE = """
Extract structured benefit data from the following Medical Benefits text. 
Target Benefits: Ambulance, Chiropractic, Annual Routine Physical, Emergency Care, Cardiac Rehab, Hearing Services.

Rules:
1. Create separate entries for sub-services (e.g., Ground Ambulance vs Air Ambulance).
2. For 'prior_auth', use 'Yes', 'No', or specific conditions mentioned.
3. For 'cost_sharing', include the exact dollar or percentage amount.
4. Capture frequency limits (e.g., 'once per year') and specific rules (e.g., 'waived if admitted') in 'important_notes'.

Return ONLY a valid JSON list of objects. Do not include markdown formatting or conversational text.

Required JSON Structure:
[
  {{
    "benefit_name": "Category",
    "service_type": "Sub-service name",
    "cost_sharing": "Amount",
    "prior_auth": "Details",
    "limitations": "Frequency limits",
    "important_notes": "Business rules/details",
    "network_info": "Provider network requirements"
  }}
]

Text:
{chunk}
"""

def extract_from_chunk(chunk):
    """
    Simpler, cleaner extraction logic that handles LLM noise by finding JSON boundaries.
    """
    payload = {
        "model": "qwen2.5-coder:1.5b",
        "prompt": PROMPT_TEMPLATE.format(chunk=chunk),
        "stream": False,
        "options": {"temperature": 0}
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        if response.status_code != 200:
            return []

        text = response.json().get("response", "").strip()

        # Find the JSON array within the text
        start = text.find('[')
        end = text.rfind(']')

        if start != -1 and end != -1:
            json_data = text[start:end+1]
            # Standardize common issues like escaped characters
            parsed = json.loads(json_data)
            return parsed if isinstance(parsed, list) else [parsed]
        
        return []

    except (json.JSONDecodeError, requests.RequestException) as e:
        print(f"⚠️ Extraction error: {e}")
        return []