import os
import re
import sqlite3
import difflib
from datetime import date, timedelta
from google import genai
from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"

# ── Database Paths ─────────────────────────────────────────────────────────────────────
DB_PATH = "sales_db.sqlite"

# ── Step 1: Create and populate the database ──────────────────────────────────
def create_database(db_path: str) -> None:
    today = date.today()

    customers = [
        (1, "Alice Johnson",  "alice@example.com",   str(today - timedelta(days=90))),
        (2, "Bob Smith",      "bob@example.com",     str(today - timedelta(days=60))),
        (3, "Carol White",    "carol@example.com",   str(today - timedelta(days=45))),
        (4, "David Brown",    "david@example.com",   str(today - timedelta(days=30))),
        (5, "Eve Davis",      "eve@example.com",     str(today - timedelta(days=15))),
    ]

    # 10 sales spread across the last 7 days
    sales = [
        (1,  1, "Laptop",      1200.00, str(today - timedelta(days=1))),
        (2,  2, "Smartphone",   800.00, str(today - timedelta(days=1))),
        (3,  3, "Headphones",   150.00, str(today - timedelta(days=2))),
        (4,  1, "Monitor",      450.00, str(today - timedelta(days=2))),
        (5,  4, "Keyboard",      75.00, str(today - timedelta(days=3))),
        (6,  5, "Mouse",         45.00, str(today - timedelta(days=3))),
        (7,  2, "Tablet",       600.00, str(today - timedelta(days=4))),
        (8,  3, "Webcam",        90.00, str(today - timedelta(days=5))),
        (9,  4, "USB Hub",       35.00, str(today - timedelta(days=6))),
        (10, 5, "SSD Drive",    220.00, str(today - timedelta(days=7))),
    ]

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS customer (
                customer_id INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL,
                email       TEXT    NOT NULL,
                join_date   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sales (
                sale_id     INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                product     TEXT    NOT NULL,
                amount      REAL    NOT NULL,
                sale_date   TEXT    NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
            );
        """)

        # Clear old sample data
        cur.execute("DELETE FROM sales")
        cur.execute("DELETE FROM customer")

        cur.executemany(
            "INSERT INTO customer VALUES (?, ?, ?, ?)", customers
        )
        cur.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?)", sales
        )
        conn.commit()

    print(f"Database created   : {db_path}")
    print(f"Customers inserted : {len(customers)}")
    print(f"Sales inserted     : {len(sales)}\n")


# ── Step 2 & 3: Ask Gemini to generate a SQL query ────────────────────────────
SCHEMA = """
Tables in the SQLite database:

customer(
    customer_id INTEGER PRIMARY KEY,
    name        TEXT,
    email       TEXT,
    join_date   TEXT          -- format: YYYY-MM-DD
)

sales(
    sale_id     INTEGER PRIMARY KEY,
    customer_id INTEGER,      -- FK → customer.customer_id
    product     TEXT,
    amount      REAL,
    sale_date   TEXT          -- format: YYYY-MM-DD
)
"""

NL_QUESTION = input("Enter your question: ")

if not NL_QUESTION.strip():
    print("Question cannot be empty.")
    raise SystemExit(1)

def get_sql_from_gemini(schema: str, question: str) -> str:
    today_str = str(date.today())

    prompt = f"""
Role:
You are an expert SQL developer working with SQLite.

Context:
Below is the schema of a SQLite database.

Database Schema:
{schema}

Task:
Convert the given natural language question into a valid SQLite SQL query.

Question:
{question}

Rules:
- Return ONLY the SQL query
- Do NOT include markdown code fences
- Do NOT include explanations or comments
- Generate ONLY one SQLite SELECT query
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE queries
- Use only the tables and columns provided in the schema
- Use SQLite compatible syntax
- Use DATE() and SQLite date arithmetic functions when needed
- Do NOT generate multiple SQL statements

Today's Date:
{today_str}
"""
    
    try:
        response = client.models.generate_content(
            model=MODEL, 
            contents=prompt,
            config={"response_mime_type": "text/plain"}
        )
        return response.text.strip()
    
    except Exception as exc:
        raise RuntimeError(f"ERROR calling Gemini API: {exc}")

def extract_sql(raw: str) -> str:
    """Strip markdown fences and return only the SQL statement."""
    # Remove ```sql ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:sql)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    # Remove any line that starts with ``` independently
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()

def contains_blocked_operation(question: str) -> bool:

    blocked_words = [
        "delete",
        "drop",
        "update",
        "insert",
        "alter",
        "truncate",
        "create"
    ]

    words = question.lower().split()

    for word in words:

        for blocked in blocked_words:

            similarity = difflib.SequenceMatcher(
                None,
                word,
                blocked
            ).ratio()

            # catches:
            # delet
            # updte
            # dropp
            # insertt

            if similarity >= 0.8:
                return True

    return False

def is_safe_sql(sql: str) -> bool:

    sql = sql.strip().lower()

    # Block multiple SQL statements
    if ";" in sql[:-1]:
        return False

    # Allow ONLY SELECT queries
    if not sql.startswith("select"):
        return False

    blocked_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create"
    ]

    if any(keyword in sql for keyword in blocked_keywords):
        return False

    return True

# ── Step 4 & 5: Execute SQL and print results ─────────────────────────────────
def execute_and_print(db_path: str, sql: str) -> None:
    print("Generated SQL Query:\n")
    print(sql)
    print()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            print(f"SQLite execution error: {exc}")
            return

    if not rows:
        print("No results for the query.")
        return

    # Pretty-print as a simple table
    col_names = rows[0].keys()
    widths    = {col: len(col) for col in col_names}
    for row in rows:
        for col in col_names:
            widths[col] = max(widths[col], len(str(row[col])))

    def fmt_row(values):
        return "  ".join(str(v).ljust(widths[c]) for c, v in zip(col_names, values))

    header  = fmt_row(col_names)
    divider = "  ".join("-" * widths[c] for c in col_names)

    print("Results:")
    print(header)
    print(divider)
    for row in rows:
        print(fmt_row([row[c] for c in col_names]))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1 – Build the database
    create_database(DB_PATH)

    # 2 – Get SQL from Gemini
    print(f"\nUser Question:\n{NL_QUESTION}\n")
    print("Asking Gemini to generate SQL …\n")

    # Block blocked operations before sending to Gemini
    if contains_blocked_operation(NL_QUESTION):

        print("Blocked database operations are not allowed.")
        raise SystemExit(1)

    raw_response = get_sql_from_gemini(SCHEMA, NL_QUESTION)

    print("Raw Gemini response:")
    print("-" * 40)
    print(raw_response)
    print("-" * 40)
    print()

    # 3 – Extract clean SQL
    sql_query = extract_sql(raw_response)

    # Validate generated SQL
    if not is_safe_sql(sql_query):

        print("Blocked SQL query detected.")
        print(f"\nGenerated Query:\n{sql_query}")
        
        raise SystemExit(1)

    # 4 & 5 – Execute and display
    execute_and_print(DB_PATH, sql_query)