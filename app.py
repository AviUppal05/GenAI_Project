import os
import io
import re
import json
import sqlite3
import difflib
import textwrap
import pathlib
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Streamlit Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Assignment",
    page_icon="🧠",
    layout="wide"
)

#st.title("🧠 GenAI Assignment Dashboard")python --version
#st.markdown("Run all 5 GenAI tasks from a single Streamlit application.")

st.title("✨ Gen AI Toolkit")
st.markdown(
    "An integrated AI workspace for prompt engineering, document intelligence, synthetic data generation, and SQL automation."
)

# ── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()

# ── Gemini Setup ────────────────────────────────────────────────────────────
client = genai.Client(api_key=api_key)
MODEL = "gemini-2.5-flash"

# ── Sidebar ────────────────────────────────────────────────────────────────
# st.sidebar.markdown("## 📚 Tasks")
st.sidebar.markdown("## 🚀 AI Tools")

selected_task = st.sidebar.selectbox(
    "",
    [
        "🧠 AI Prompt Studio",
        "💬 Chat with AI",
        "📊 Data Generator",
        "📄 PDF Intelligence",
        "🗄️ AI SQL Assistant"
    ]
)

# selected_task = st.sidebar.selectbox(
#     "",
#     [
#         "Task 1 - Prompt Engineering",
#         "Task 2 - LLM Chat",
#         "Task 3 - Data Augmentation",
#         "Task 4 - Data Querying",
#         "Task 5 - Data Extraction"
#     ]
# )

# ── Sidebar ────────────────────────────────────────────────────────────────
# st.sidebar.title("Tasks")
# selected_task = st.sidebar.radio(
#     "Select a Task",
#     [
#         "Task 1 - Prompt Engineering",
#         "Task 2 - LLM Chat",
#         "Task 3 - Data Augmentation",
#         "Task 4 - Data Querying",
#         "Task 5 - Data Extraction"
#     ]
# )


# =============================================================================
# TASK 1 — PROMPT ENGINEERING
# =============================================================================
# if selected_task == "Task 1 - Prompt Engineering":
if selected_task == "🧠 AI Prompt Studio":

    # st.header("Task 1 — Prompt Engineering")
    st.header("🧠 AI Prompt Studio")

    prompts = {
        "SQL Query Optimisation": textwrap.dedent("""
            ## Context
            You are assisting a data engineering team that maintains a cloud data warehouse.

            ## Role
            Act as a senior SQL performance expert.

            ## Task
            Optimise the following SQL query.

            SELECT * FROM fact_sales;

            ## Constraints
            - Use standard SQL
            - Explain optimisations
        """),

        "Real-Time Pipeline Design": textwrap.dedent("""
            Design a real-time AWS data pipeline for clickstream events.
        """),

        "Data Quality Checks": textwrap.dedent("""
            Generate PostgreSQL data quality checks for a staging table.
        """)
    }

    selected_prompt = st.selectbox(
        "Choose Prompt",
        list(prompts.keys())
    )

    if st.button("Generate Response"):

        prompt = prompts[selected_prompt]

        st.subheader("Prompt")
        st.code(prompt)

        with st.spinner("Generating response..."):

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"response_mime_type": "text/plain"}
            )

        st.subheader("Gemini Response")
        st.write(response.text)


# =============================================================================
# TASK 2 — LLM CHAT / JSON OUTPUT
# =============================================================================
#elif selected_task == "Task 2 - LLM Chat":
elif selected_task == "💬 Chat with AI":

    #st.header("Task 2 — Structured JSON Response")
    st.header("💬 Chat with AI")

    user_activity = st.text_area(
        "Enter User Activity Log",
        value="""
User activity log:
- User A logged in and purchased a laptop worth $1200
- User B logged in but did not make any purchase
- User C purchased a phone worth $800
""",
        height=200
    )

    if st.button("Generate JSON"):

        prompt = f"""
You are a data analyst.

Return ONLY valid JSON.

The JSON must contain:
- summary
- total_users
- purchasing_users
- total_revenue
- insights

Activity Log:
{user_activity}
"""

        with st.spinner("Generating JSON..."):

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )

        raw_text = response.text.strip()

        try:
            parsed = json.loads(raw_text)

            st.subheader("JSON Output")
            st.json(parsed)

        except Exception as exc:
            st.error(f"JSON Parsing Error: {exc}")
            st.code(raw_text)


# =============================================================================
# TASK 3 — DATA AUGMENTATION
# =============================================================================
#elif selected_task == "Task 3 - Data Augmentation":
elif selected_task == "📊 Data Generator":

    #st.header("Task 3 — CSV Data Augmentation")
    st.header("📊 Data Generator")

    uploaded_csv = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_csv:

        df_original = pd.read_csv(uploaded_csv)

        st.subheader("Original Dataset")
        st.dataframe(df_original, use_container_width=True)

        if st.button("Generate More Rows"):

            csv_content = df_original.to_csv(index=False)

            prompt = f"""
Role:
You are a data generation assistant.

Task:
Generate exactly 10 more realistic rows.

Rules:
- Maintain same columns
- Generate unique rows
- Return ONLY CSV rows
- No markdown

Existing Data:
{csv_content}
"""

            with st.spinner("Generating augmented rows..."):

                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config={"response_mime_type": "text/plain"}
                )

            raw_text = response.text.strip()

            if raw_text.startswith("```"):
                raw_text = raw_text.replace("```csv", "")
                raw_text = raw_text.replace("```", "")
                raw_text = raw_text.strip()

            try:

                df_new = pd.read_csv(
                    io.StringIO(raw_text),
                    header=None,
                    names=df_original.columns
                )

                final_df = pd.concat(
                    [df_original, df_new],
                    ignore_index=True
                )

                st.subheader("Augmented Dataset")
                st.dataframe(final_df, use_container_width=True)

                csv_data = final_df.to_csv(index=False).encode("utf-8")
                
                # Get uploaded filename
                original_filename = uploaded_csv.name
                
                # Remove extension
                base_name = os.path.splitext(original_filename)[0]
                
                # Create dynamic augmented filename
                download_filename = f"augmented_{base_name}.csv"
                
                st.download_button(
                    label="Download Augmented CSV",
                    data=csv_data,
                    file_name=download_filename,
                    mime="text/csv"
                )

                # st.download_button(
                #     label="Download Augmented CSV",
                #     data=csv_data,
                #     file_name="augmented_customers.csv",
                #     mime="text/csv"
                # )

            except Exception as exc:
                st.error(f"CSV Parsing Error: {exc}")
                st.code(raw_text)


# =============================================================================
# TASK 4 — PDF DATA QUERYING
# =============================================================================
#elif selected_task == "Task 4 - Data Querying":
elif selected_task == "📄 PDF Intelligence":

    #st.header("Task 4 — PDF Question Answering")
    st.header("📄 PDF Intelligence")

    uploaded_pdf = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"]
    )

    question = st.text_input(
        "Enter your question",
        placeholder="Example: Summarize this document"
    )

    if uploaded_pdf and question:

        if st.button("Ask Gemini"):

            prompt = f"""
Context:
You are given a PDF document.

Task:
Answer the question based ONLY on the document.

Constraints:
- Do not hallucinate
- Keep answer concise

Question:
{question}
"""

            with st.spinner("Analyzing PDF..."):

                response = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Part.from_bytes(
                            data=uploaded_pdf.read(),
                            mime_type="application/pdf"
                        ),
                        prompt
                    ],
                    config={"response_mime_type": "text/plain"}
                )

            st.subheader("Answer")
            st.write(response.text)


# =============================================================================
# TASK 5 — NL TO SQL
# =============================================================================
#elif selected_task == "Task 5 - Data Extraction":
elif selected_task == "🗄️ AI SQL Assistant":

    #st.header("Task 5 — Natural Language to SQL")
    st.header("🗄️ AI SQL Assistant")

    DB_PATH = "sales_db.sqlite"

    # ── Create Database ─────────────────────────────────────────────────────
    def create_database(db_path: str):

        today = date.today()

        customers = [
            (1, "Alice Johnson", "alice@example.com", str(today - timedelta(days=90))),
            (2, "Bob Smith", "bob@example.com", str(today - timedelta(days=60))),
            (3, "Carol White", "carol@example.com", str(today - timedelta(days=45))),
        ]

        sales = [
            (1, 1, "Laptop", 1200.00, str(today - timedelta(days=1))),
            (2, 2, "Phone", 800.00, str(today - timedelta(days=2))),
            (3, 1, "Tablet", 500.00, str(today - timedelta(days=3))),
        ]

        with sqlite3.connect(db_path) as conn:

            cur = conn.cursor()

            cur.executescript("""
                CREATE TABLE IF NOT EXISTS customer (
                    customer_id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    join_date TEXT
                );

                CREATE TABLE IF NOT EXISTS sales (
                    sale_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    product TEXT,
                    amount REAL,
                    sale_date TEXT
                );
            """)

            cur.execute("DELETE FROM customer")
            cur.execute("DELETE FROM sales")

            cur.executemany(
                "INSERT INTO customer VALUES (?, ?, ?, ?)",
                customers
            )

            cur.executemany(
                "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
                sales
            )

            conn.commit()


    if not os.path.exists(DB_PATH):
        create_database(DB_PATH)


    # ── Safety Functions ────────────────────────────────────────────────────
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

                if similarity >= 0.8:
                    return True

        return False


    def is_safe_sql(sql: str) -> bool:

        sql = sql.strip().lower()

        if ";" in sql[:-1]:
            return False

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


    # ── Schema ──────────────────────────────────────────────────────────────
    SCHEMA = """
customer(customer_id, name, email, join_date)
sales(sale_id, customer_id, product, amount, sale_date)
"""

    user_question = st.text_input(
        "Enter your question",
        placeholder="Example: Highest sales in last 3 days"
    )

    if st.button("Generate SQL"):

        if not user_question.strip():
            st.warning("Question cannot be empty.")
            st.stop()

        if contains_blocked_operation(user_question):
            st.error("Blocked database operations are not allowed.")
            st.stop()

        today_str = str(date.today())

        prompt = f"""
Role:
You are an expert SQL developer.

Database Schema:
{SCHEMA}

Task:
Convert the natural language question into SQLite SQL.

Question:
{user_question}

Rules:
- Return ONLY SQL query
- Generate ONLY SELECT query
- No markdown
- No explanation

Today's Date:
{today_str}
"""

        with st.spinner("Generating SQL query..."):

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"response_mime_type": "text/plain"}
            )

        sql_query = response.text.strip()

        st.subheader("Generated SQL")
        st.code(sql_query, language="sql")

        if not is_safe_sql(sql_query):
            st.error("Blocked SQL query detected.")
            st.stop()

        try:

            with sqlite3.connect(DB_PATH) as conn:
                result_df = pd.read_sql_query(sql_query, conn)

            st.subheader("Query Results")

            if result_df.empty:
                st.info("No results found.")
            else:
                st.dataframe(result_df, use_container_width=True)

        except Exception as exc:
            st.error(f"SQL Execution Error: {exc}")