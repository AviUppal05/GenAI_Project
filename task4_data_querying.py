import os
import pathlib
from google import genai
from google.genai import types
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

# ── Load PDF Document ───────────────────────────────────────────────
filepath = pathlib.Path('sample.pdf')

print(f"\nDocument loaded: {filepath}\n")

# ── Questions ───────────────────────────────────────────────
QUESTIONS = [
    "Q1: Please provide a summary of the PDF document.",
    "Q2: What are the key topics discussed in the document?"
]

# ── Questions ───────────────────────────────────────────────
# QUESTIONS = [
#     {
#         "label": "Q1 – Summary",
#         "text": "Please provide a summary of the PDF document."
#     },
#     {
#         "label": "Q2 – Key Topics",
#         "text": "What are the key topics discussed in the document?"
#     }
# ]

# ── Helper Function ──────────────────────────────────────────────────────────
def ask_question(question: str) -> str:

    prompt = f"""
Context:
You are given a PDF document.

Role:
Act as an intelligent document analyst.

Task:
Answer the question based ONLY on the provided document.

Constraints:
- Do not hallucinate
- If information is not available, say "Not available in document"
- Keep the answer concise and accurate

Question:
{question}
"""
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type="application/pdf",
                ),
                prompt
            ]
        )

        return response.text.strip()

    except Exception as exc:
        return f"ERROR calling Gemini API: {exc}" 
    

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    separator = "=" * 72

    for question in QUESTIONS:
        
        print(f"\n{question}\n")
        
        answer = ask_question(question)
        
        print(f"Answer:\n{answer}\n")
    
    print(separator)



# if __name__ == "__main__":

#     separator = "=" * 72

#     for q in QUESTIONS:

#         print(separator)
#         print(f"  {q['label']}")
#         print(separator)

#         print(f"\nQuestion:\n  {q['text']}\n")

#         answer = ask_question(q["text"])

#         print(f"Answer:\n{answer}\n")

#     print(separator)