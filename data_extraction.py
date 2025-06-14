import json
import re

def extract_json(response_content):
    """
    Extracts a JSON object from a potentially noisy LLM response.
    Handles:
    - ```json ... ```
    - Just {...}
    - Improper whitespace/control characters
    """
    # 1. Prefer content inside ```json ``` if present
    match = re.search(r"```json\s*(.*?)\s*```", response_content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 2. Otherwise, try to find the first {...} block
        match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            # 3. Fallback if nothing matches
            print("❌ No valid JSON block found.")
            return {"answer": "Image not processed (no JSON found)"}

    # 4. Remove dangerous control characters
    json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)

    # 5. Try parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("❌ JSON decode error:", e)
        print("⚠️ Problematic JSON string:", json_str)
        return {"answer": "Image not processed (malformed JSON)"}
