import os
import io
import pandas as pd
from google import genai
from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()

# Load API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file.")

# Initialize Gemini client
client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"

INPUT_CSV  = "customers.csv"
OUTPUT_CSV = "augmented_customers.csv"

# ── Step 1: Read CSV ─────────────────────────────────────────────────────
print("Reading original data from customers.csv …")
df_original = pd.read_csv(INPUT_CSV)
print(df_original.to_string(index=False))

# ── Step 2: Build prompt ──────────────────────────────────────────────────────
csv_content = df_original.to_csv(index=False)

PROMPT = f"""
Role:
You are a data generation assistant.

Context:
Below is a CSV dataset of customers. 

Task:
Generate exactly 10 MORE similar, realistic rows 

Rules:
- follow the SAME column order and same schema as the original dataset given as input.
- customer_id or similar column values must start from {len(df_original) + 1} and increment sequentially by 1.
- Names, emails, and cities must be realistic but fictional.
- purchase_amount must be a positive number with 2 decimal places.
- Ensure generated rows are unique.
- Be sure to not any duplicate data records.
- Return ONLY the 10 data rows as plain CSV text.
- Do NOT include a header row.
- Do NOT include markdown fences, explanations, or any extra text.

Existing data (for style reference):
{csv_content}
"""

# ── Step 3: Call Gemini ───────────────────────────────────────────────────────
print("\nRequesting 10 augmented rows from Gemini …\n")
try:
    response = client.models.generate_content(
        model=MODEL, 
        contents=PROMPT, 
        config={"response_mime_type": "text/plain"
        }
    )

    raw_text = response.text.strip()
    
except Exception as exc:
    print(f"ERROR calling Gemini API: {exc}")
    raise SystemExit(1)

# ── Step 4: Strip accidental markdown fences if present ─────────────────────────────────────────
if raw_text.startswith("```"):
    lines = raw_text.splitlines()

    #Remove opening fence (```csv or ```)
    if lines[0].startswith("```"):
        lines = lines[1:]
    
    #Remove closing fence (```)
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    raw_text = "\n".join(lines).strip()

print("\nRaw CSV response from Gemini:\n")
print("-" * 40)
print(raw_text)
print("-" * 40)

# ── Step 5: Convert generated CSV text into DataFrame ───────────────────────
try:
    df_new = pd.read_csv(
        io.StringIO(raw_text),
        header=None,
        names=df_original.columns
    )

    # Combine with original dataset
    final_df = pd.concat([df_original, df_new], ignore_index=True)

    # Print final augmented dataset
    print("\nFinal Dataset:\n")
    print("-" * 60)

    print(final_df.to_string(index=False))

    print("-" * 60)

    # Save augmented dataset
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAugmented dataset saved as: {OUTPUT_CSV}\n")

except Exception as e:
    print("Error parsing generated CSV:", e)