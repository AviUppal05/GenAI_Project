import os
import json
from google import genai
from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()

# Load API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file.")

# Initialize Gemini client
client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"

# ── User activity log ────────────────────────────────────────────────────────
USER_ACTIVITY = """
User activity log:
- User A logged in and purchased a laptop worth $1200
- User B logged in but did not make any purchase
- User C purchased a phone worth $800
"""

# ── Prompt instructing Gemini to return strict JSON ──────────────────────────
PROMPT = f"""
You are a data analyst. Analyse the following user activity log and return a
summary as JSON.

IMPORTANT: Return ONLY valid JSON. No markdown code fences, no extra text,
no explanations — just the raw JSON object.

The JSON must have exactly these keys:
  "summary"          – a one-sentence plain-English summary of the activity
  "total_users"      – integer count of unique users mentioned
  "purchasing_users" – integer count of users who made at least one purchase
  "total_revenue"    – numeric total of all purchase amounts (numbers only, no $)
  "insights"         – a list of 2 short insight strings

Activity log:
{USER_ACTIVITY.strip()}
"""
#Another way to write the prompt
# PROMPT = f"""
# Role:
# You are a data analyst.

# Task:
# Analyze the following user activity log and return structured insights as JSON.

# Constraints:
# - Keep the response concise
# - Return ONLY valid JSON
# - No markdown code fences
# - No explanations
# - No extra text

# Required Output Format:
# {{
#   "summary": "...",
#   "total_users": number,
#   "purchasing_users": number,
#   "total_revenue": number,
#   "insights": [
#     "...",
#     "..."
#   ]
# }}

# Activity Log:
# {USER_ACTIVITY.strip()}
# """

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sending user activity log to Gemini...\n")

# Call Gemini 
    try:
        response = client.models.generate_content(
            model=MODEL, 
            contents=PROMPT, 
            config={"response_mime_type": "application/json"
            }
        )
        raw_text = response.text.strip()

    except Exception as exc:
        print(f"ERROR calling Gemini API: {exc}")
        raise SystemExit(1)

# Strip any accidental markdown fences the model might still add
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()

        # Remove opening fence (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]

        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_text = "\n".join(lines).strip()


# Convert to JSON
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"\nERROR: Could not parse response as JSON.\nDetail: {exc}")
        raise SystemExit(1)
 
### JSON Response
    print("\nJSON Response:")
    print("=" * 40)
    print(json.dumps(parsed, indent=2))
    print("=" * 40)