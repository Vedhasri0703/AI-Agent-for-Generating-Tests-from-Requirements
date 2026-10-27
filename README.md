# 🧪 Requirement-Based Test Generation Agent

A focused agentic AI application that turns natural-language requirements into structured, traceable test cases.

## What it does

1. **Accepts requirements** as pasted text or uploaded TXT / DOCX / PDF
2. **Analyzes ambiguity** — identifies unclear terms, missing rules, conflicting conditions
3. **Asks targeted clarification questions** — only what matters for test accuracy
4. **Extracts acceptance criteria** — assigns stable REQ-001, REQ-002, … IDs
5. **Generates test cases** — positive, negative, boundary, validation & error-handling
6. **Validates traceability** — every test must link to a real requirement ID
7. **Exports** results as CSV or JSON

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key
```

Or enter it directly in the sidebar when the app is running.

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
├── app.py                  # Main Streamlit application
├── src/
│   ├── models.py           # Pydantic data models (AgentState, TestCase, etc.)
│   ├── llm_tools.py        # LLM-powered tool functions
│   ├── agent.py            # Workflow orchestration (state machine)
│   ├── storage.py          # SQLite persistence + CSV/JSON export
│   └── ui_components.py    # Reusable UI components
├── data/                   # SQLite database (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## Technology Stack

| Area | Choice |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| LLM | OpenAI via LangChain |
| Schema validation | Pydantic v2 |
| Document parsing | python-docx, pypdf |
| Persistence | SQLite |
| Export | CSV, JSON |

## Test Case Schema

Each generated test case includes:

| Field | Description |
|---|---|
| `test_id` | Unique ID (TC-001, TC-002, …) |
| `requirement_ids` | One or more REQ-XXX IDs |
| `title` | Short descriptive name |
| `category` | positive / negative / boundary / validation / error-handling |
| `preconditions` | Required state before the test |
| `test_data` | Input values |
| `steps` | Actions in execution order |
| `expected_result` | Observable outcome |
| `priority` | high / medium / low |
| `review_status` | draft / reviewed / needs changes |

## Limitations

- Does not execute tests or inspect source code
- LLM output may require human review — always check generated criteria
- Traceability validates that links *exist*, not that tests are semantically correct
- Requirements with sensitive data are sent to the OpenAI API; review their privacy policy
