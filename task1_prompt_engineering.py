import os
import textwrap
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

# ── Prompt 1 – SQL Query Optimisation ────────────────────────────────────────
PROMPT_1 = textwrap.dedent("""
    ## Context
    You are assisting a data engineering team that maintains a cloud data warehouse
    (Snowflake). The central fact table `fact_sales` contains ~500 million rows
    partitioned by `sale_date` with a clustering key on (sale_date, region_id).
    The table has frequently-queried foreign keys to dimension tables.

    ## Role
    Act as a senior data engineer and SQL performance expert with deep experience
    in large-scale analytical workloads.

    ## Task
    Review the following query and rewrite it to maximise performance on Snowflake.
    Explain every optimisation you apply.

    ```sql
    SELECT
        c.customer_name,
        SUM(f.amount) AS total_sales,
        COUNT(*) AS num_transactions
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_id = c.customer_id
    WHERE YEAR(f.sale_date) = 2024
      AND f.region_id IN (SELECT region_id FROM dim_region WHERE country = 'US')
    GROUP BY c.customer_name
    ORDER BY total_sales DESC;
    ```

    ## Constraints
    - Do not change the business logic or the output columns.
    - Use only standard Snowflake SQL syntax.
    - Avoid full table scans wherever possible.
    - Keep the rewritten query under 30 lines.

    ## Format
    Return your answer in two clearly labelled sections:
    1. **Optimised Query** – the rewritten SQL block.
    2. **Optimisations Applied** – a numbered list explaining each change.

    ## Examples
    Example of a good optimisation note:
    "1. Replaced correlated sub-query with a JOIN to eliminate repeated scans."
""").strip()

# ── Prompt 2 – Real-Time Data Pipeline Design ────────────────────────────────
PROMPT_2 = textwrap.dedent("""
    ## Context
    A retail company generates ~50,000 clickstream events per second from its
    e-commerce platform. Events must be ingested, transformed, and available for
    analytics within 5 seconds (near-real-time SLA). The company uses AWS.

    ## Role
    Act as a cloud data architect specialising in streaming data pipelines and
    AWS managed services.

    ## Task
    Design a complete end-to-end real-time data ingestion pipeline that:
    - Ingests raw JSON events from an HTTP endpoint
    - Validates and deduplicates events
    - Applies lightweight transformations (flatten nested fields, parse timestamps)
    - Lands clean data in a queryable destination within 5 seconds

    ## Constraints
    - Use only AWS managed services (no self-managed Kafka or Flink clusters).
    - Must handle at least 50,000 events/second at peak load.
    - Cost should be optimised; avoid over-provisioned fixed-capacity services.
    - Provide fault-tolerance and at-least-once delivery guarantees.
    - Do not include machine-learning components.

    ## Format
    Structure your answer as:
    1. **Architecture Diagram (text)** – an ASCII or textual flow diagram.
    2. **Component Descriptions** – brief explanation of each service chosen.
    3. **Data Flow Steps** – numbered sequence from event source to destination.
    4. **Trade-offs & Risks** – 2–3 bullet points.

    ## Examples
    A sample service selection:
    "API Gateway → Kinesis Data Streams → Lambda → Kinesis Firehose → S3 → Athena"
""").strip()

# ── Prompt 3 – Data Quality Checks for a Staging Table ───────────────────────
PROMPT_3 = textwrap.dedent("""
    ## Context
    A nightly ETL job loads raw CSV data from an SFTP server into a staging table
    called `stg_orders` in PostgreSQL before it is promoted to the production
    `orders` table. The staging table has columns:
    order_id (INT), customer_id (INT), product_code (VARCHAR), quantity (INT),
    unit_price (NUMERIC), order_date (DATE), status (VARCHAR).

    ## Role
    Act as a data quality engineer responsible for ensuring data accuracy,
    completeness, and consistency before data reaches production.

    ## Task
    Generate a comprehensive set of data quality checks for the `stg_orders`
    staging table. Checks should cover:
    - Completeness (nulls, missing values)
    - Uniqueness (duplicate detection)
    - Validity (data types, allowed values, ranges)
    - Referential integrity (foreign key-like checks against other tables)
    - Timeliness (stale or future-dated records)

    ## Constraints
    - Express all checks as executable PostgreSQL SQL queries.
    - Each check must return 0 rows if the data is clean (zero-result assertion style).
    - Provide at least 2 checks per category.
    - Do not use stored procedures or proprietary extensions.

    ## Format
    For each check provide:
    - **Check Name**: short descriptive title
    - **Category**: (Completeness / Uniqueness / Validity / Referential Integrity / Timeliness)
    - **SQL**: the query
    - **Failure Meaning**: one sentence explaining what a non-zero result indicates

    ## Examples
    Check Name: Null Order IDs
    Category: Completeness
    SQL: SELECT * FROM stg_orders WHERE order_id IS NULL;
    Failure Meaning: Rows exist with no order identifier, which will break the primary key constraint.
""").strip()

# ── Helper – send prompt and print result ─────────────────────────────────────
def send_and_print(label: str, prompt: str) -> None:
    separator = "=" * 72
    print(f"\n{separator}")
    print(f"  PROMPT: {label}")
    print(separator)
    print("\n[Prompt Text]\n")
    print(prompt)
    print("\n[Gemini Response]\n")
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        print(response.text)
    except Exception as exc:
        print(f"ERROR calling Gemini API: {exc}")
    print(separator)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    send_and_print("SQL Query Optimisation for a Large Fact Table", PROMPT_1)
    send_and_print("Designing a Data Pipeline for Real-Time Ingestion", PROMPT_2)
    send_and_print("Data Quality Checks for a Staging Table", PROMPT_3)