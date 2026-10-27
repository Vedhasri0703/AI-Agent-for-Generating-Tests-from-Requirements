"""
LLM-powered tool functions for the Test Generation Agent.
Uses Groq only — controlled by GROQ_API_KEY and GROQ_MODEL env vars.
"""

from __future__ import annotations
import json
import os
import re
import warnings
from typing import List
from dotenv import load_dotenv

load_dotenv(override=False)

warnings.filterwarnings("ignore", message=".*non-supported Python version.*")

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.models import (
    AmbiguityAnalysis,
    ClarificationQuestion,
    AcceptanceCriterion,
    TestCase,
    ValidationResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry — Groq only (verified Oct 2026, free developer plan)
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODELS = {
    "openai/gpt-oss-120b": "GPT OSS 120B  ✅ recommended (500 t/s)",
    "openai/gpt-oss-20b":  "GPT OSS 20B   ⚡ fastest    (1000 t/s)",
    "qwen/qwen3.8-27b":    "Qwen 3.8 27B  🧠 capable    (450 t/s)",
}

# Kept as empty dict so any remaining import reference doesn't break
GEMINI_MODELS = {}
OPENAI_MODELS = {}


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory — Groq only
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return a Groq LangChain chat model. temperature=0 for deterministic JSON."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction & repair helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_block(text: str) -> dict | list:
    """
    Robustly extract JSON from LLM output.

    Handles:
    - ```json ... ``` and ``` ... ``` fenced blocks
    - ```python ... ``` blocks (ignored — model generated code instead of JSON)
    - Leading prose before the first [ or {
    - Trailing commas, // comments, /* */ comments
    - Python None/True/False literals
    - Truncated arrays (brace-matching recovery)
    """
    # Step 1: strip non-JSON fenced code blocks (python, typescript, etc.)
    # If the whole response is inside a ```python block, there's no JSON — fail fast
    only_code_block = re.fullmatch(
        r"\s*```(?!json)(\w*)\s*[\s\S]+?```\s*", text, re.IGNORECASE
    )
    if only_code_block:
        raise ValueError(
            f"LLM returned {only_code_block.group(1) or 'code'} instead of JSON. "
            f"Raw text:\n{text[:400]}"
        )

    # Step 2: extract content from ```json block if present
    json_block = re.search(r"```json\s*([\s\S]+?)```", text, re.IGNORECASE)
    if json_block:
        raw = json_block.group(1).strip()
    else:
        # Step 3: strip any remaining non-json ``` blocks, keep plain text
        raw = re.sub(r"```\w*\s*[\s\S]*?```", "", text, flags=re.IGNORECASE).strip()

    # Step 4: find the first JSON structure character
    first_brace   = raw.find("{")
    first_bracket = raw.find("[")
    if first_brace == -1 and first_bracket == -1:
        raise ValueError(
            f"No JSON object or array found in LLM output.\nRaw text:\n{raw[:500]}"
        )
    start = (
        first_bracket if first_brace == -1 else
        first_brace   if first_bracket == -1 else
        min(first_brace, first_bracket)
    )
    raw = raw[start:]

    # Step 5: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Step 6: clean and retry
    cleaned = _clean_llm_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 7: recover truncated array
    recovered = _recover_truncated_array(cleaned)
    if recovered:
        return recovered

    raise ValueError(
        f"Could not parse LLM JSON after all recovery attempts.\n"
        f"First 400 chars:\n{raw[:400]}"
    )


def _clean_llm_json(text: str) -> str:
    """Remove common LLM JSON mistakes."""
    text = re.sub(r'//[^\n"]*\n', '\n', text)           # // comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # /* */ comments
    text = re.sub(r',\s*([\}\]])', r'\1', text)           # trailing commas
    text = re.sub(r'\bNone\b',  'null',  text)
    text = re.sub(r'\bTrue\b',  'true',  text)
    text = re.sub(r'\bFalse\b', 'false', text)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def _recover_truncated_array(text: str) -> list | None:
    """Salvage complete objects from a truncated JSON array."""
    if not text.strip().startswith("["):
        return None
    objects, depth, start, in_str, escape = [], 0, None, False, False
    for i, ch in enumerate(text):
        if escape:           escape = False; continue
        if ch == "\\" and in_str: escape = True; continue
        if ch == '"':        in_str = not in_str; continue
        if in_str:           continue
        if ch == "{":
            if depth == 0:   start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(_clean_llm_json(text[start:i+1])))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects if objects else None


def _invoke_with_json_retry(
    llm: BaseChatModel,
    messages: list,
    label: str = "LLM call",
    max_retries: int = 2,
) -> dict | list:
    """
    Invoke the LLM and parse JSON.
    On parse failure, send a correction message asking the model to fix its output.
    """
    history = list(messages)
    last_error = None

    for attempt in range(max_retries + 1):
        response = llm.invoke(history)
        raw = response.content

        try:
            return _parse_json_block(raw)
        except ValueError as e:
            last_error = e
            if attempt < max_retries:
                # Ask the model to correct itself
                correction = (
                    "Your previous response was not valid JSON. "
                    "You must respond with ONLY a raw JSON object or array — "
                    "no Python code, no markdown, no prose, no code fences. "
                    "Output ONLY the JSON now."
                )
                # Append the bad response and correction as a new turn
                history.append(HumanMessage(content=f"[Previous output]\n{raw}\n\n{correction}"))

    raise ValueError(
        f"{label} failed after {max_retries + 1} attempts.\n"
        f"Last error: {last_error}\n"
        f"Last raw output (first 400 chars):\n{raw[:400]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# System prompt prefix — forces coding models away from code generation
# ─────────────────────────────────────────────────────────────────────────────

_JSON_ONLY_PREFIX = (
    "CRITICAL INSTRUCTION: You are a JSON API endpoint, not a code generator. "
    "Your ONLY job is to return a valid JSON object or array as specified below. "
    "Do NOT write Python, JavaScript, or any other code. "
    "Do NOT explain your output. Do NOT add markdown fences. "
    "Output RAW JSON and nothing else.\n\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: analyze_requirement_ambiguity
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_SCHEMA = '''{
  "has_ambiguity": true,
  "issues": ["vague term in rule X", "missing condition for Y"],
  "questions": [
    {"question": "...", "reason": "..."}
  ],
  "summary": "One sentence summary."
}'''

AMBIGUITY_SYSTEM = _JSON_ONLY_PREFIX + f"""You are a senior QA analyst identifying ambiguities in software requirements.

Return this exact JSON schema (no other text):
{AMBIGUITY_SCHEMA}

Rules:
- has_ambiguity: true only if there are questions that would change the test cases.
- Maximum 5 questions.
- Only ask questions whose answers materially affect test behavior.
- If requirements are fully clear, return has_ambiguity: false and empty questions list.
"""


def analyze_requirement_ambiguity(
    requirements: str,
    clarification_answers: dict | None = None,
) -> AmbiguityAnalysis:
    llm = _get_llm(temperature=0.0)
    context = f"Requirements to analyse:\n{requirements}"
    if clarification_answers:
        context += "\n\nAlready answered:\n" + "\n".join(
            f"Q: {q}\nA: {a}" for q, a in clarification_answers.items()
        )
    messages = [
        SystemMessage(content=AMBIGUITY_SYSTEM),
        HumanMessage(content=context),
    ]
    data = _invoke_with_json_retry(llm, messages, label="ambiguity analysis")

    return AmbiguityAnalysis(
        has_ambiguity=data.get("has_ambiguity", False),
        issues=data.get("issues", []),
        questions=[
            ClarificationQuestion(question=q["question"], reason=q["reason"])
            for q in data.get("questions", [])
        ],
        summary=data.get("summary", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: extract_acceptance_criteria
# ─────────────────────────────────────────────────────────────────────────────

CRITERIA_SCHEMA = '''[
  {
    "req_id": "REQ-001",
    "title": "Short title",
    "criterion": "Explicit testable statement",
    "source_text": "Exact quote from requirements",
    "status": "confirmed",
    "assumption_note": null
  }
]'''

CRITERIA_SYSTEM = _JSON_ONLY_PREFIX + f"""You are a QA analyst extracting acceptance criteria from requirements.

Return a JSON array matching this schema exactly (no other text):
{CRITERIA_SCHEMA}

Rules:
- Use sequential IDs: REQ-001, REQ-002, etc.
- One criterion per distinct testable behavior.
- status must be exactly one of: "confirmed", "assumed", "unresolved"
- assumption_note: null for confirmed; short explanation for assumed/unresolved.
- All field values must be short strings — one sentence each.
"""


def extract_acceptance_criteria(
    requirements: str,
    clarification_answers: dict | None = None,
    assumptions: list[str] | None = None,
) -> List[AcceptanceCriterion]:
    llm = _get_llm(temperature=0.0)
    context = f"Requirements:\n{requirements}"
    if clarification_answers:
        context += "\n\nClarifications:\n" + "\n".join(
            f"Q: {q}\nA: {a}" for q, a in clarification_answers.items()
        )
    if assumptions:
        context += "\n\nAssumptions:\n" + "\n".join(f"- {a}" for a in assumptions)

    messages = [
        SystemMessage(content=CRITERIA_SYSTEM),
        HumanMessage(content=context),
    ]
    data = _invoke_with_json_retry(llm, messages, label="criteria extraction")
    return [AcceptanceCriterion(**item) for item in data]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: generate_test_cases
# ─────────────────────────────────────────────────────────────────────────────

TEST_SCHEMA = '''[
  {
    "test_id": "TC-001",
    "requirement_ids": ["REQ-001"],
    "title": "Short name",
    "category": "positive",
    "preconditions": "State before test",
    "test_data": "Input values",
    "steps": ["Step 1", "Step 2"],
    "expected_result": "What should happen",
    "priority": "high",
    "review_status": "draft",
    "notes": null
  }
]'''

TEST_GEN_SYSTEM = _JSON_ONLY_PREFIX + f"""You are a QA engineer generating test cases.

Return a JSON array matching this schema exactly (no other text):
{TEST_SCHEMA}

Test categories to use:
- positive      : happy path, feature works as expected
- negative      : invalid input or unauthorised action
- boundary      : values at the edge of numeric thresholds
- validation    : input format or type checks
- error-handling: errors, timeouts, unexpected states

Rules:
- IDs: use the TC number provided in the prompt (e.g. start from TC-003).
- requirement_ids: only reference REQ IDs from the criteria given to you.
- Generate at least: 1 positive + 1 negative per criterion.
- Add boundary tests only when numeric thresholds exist.
- steps: 2 to 4 items, each one short action sentence.
- All string values: one sentence maximum.
- notes: null unless there is a specific note to add.
"""


def generate_test_cases(
    criteria: List[AcceptanceCriterion],
    requirements: str,
) -> List[TestCase]:
    """
    Generate test cases in batches of 3 criteria per LLM call
    to stay well within output token limits.
    """
    llm = _get_llm(temperature=0.0)
    all_tests: List[TestCase] = []
    tc_counter = 1
    batch_size = 3

    for batch_start in range(0, len(criteria), batch_size):
        batch = criteria[batch_start: batch_start + batch_size]

        criteria_lines = [
            f"{c.req_id}: {c.criterion}"
            + (f" [assumed: {c.assumption_note}]" if c.assumption_note else "")
            for c in batch
        ]
        context = (
            f"Requirements (summary):\n{requirements[:600]}\n\n"
            f"Acceptance criteria to cover:\n"
            + "\n".join(criteria_lines)
            + f"\n\nStart test IDs from TC-{tc_counter:03d}."
        )

        messages = [
            SystemMessage(content=TEST_GEN_SYSTEM),
            HumanMessage(content=context),
        ]

        try:
            batch_data = _invoke_with_json_retry(
                llm, messages,
                label=f"test gen batch {batch_start // batch_size + 1}",
            )
            batch_tests = [TestCase(**item) for item in batch_data]
            all_tests.extend(batch_tests)
            tc_counter += len(batch_tests)
        except Exception as e:
            print(f"[WARN] Batch {batch_start // batch_size + 1} failed: {e}")
            continue

    if not all_tests:
        raise ValueError(
            "Test case generation produced no results. "
            "The LLM may have returned code or prose instead of JSON. "
            "Try switching to a different model in the sidebar."
        )

    return all_tests


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: validate_traceability  (deterministic — no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def validate_traceability(
    criteria: List[AcceptanceCriterion],
    test_cases: List[TestCase],
) -> ValidationResult:
    errors:        list[str] = []
    warn:          list[str] = []
    invalid_links: list[str] = []
    uncovered:     list[str] = []

    valid_ids = {c.req_id for c in criteria}

    # Unique test IDs
    seen: set[str] = set()
    for tc in test_cases:
        if tc.test_id in seen:
            errors.append(f"Duplicate test_id: {tc.test_id}")
        seen.add(tc.test_id)

    # All linked REQ IDs exist
    for tc in test_cases:
        for rid in tc.requirement_ids:
            if rid not in valid_ids:
                errors.append(f"Invalid requirement link: {tc.test_id} → {rid}")
                invalid_links.append(f"{tc.test_id} → {rid}")

    # Every criterion has at least one test
    covered = {rid for tc in test_cases for rid in tc.requirement_ids}
    for crit in criteria:
        if crit.req_id not in covered:
            if crit.status == "unresolved":
                warn.append(f"{crit.req_id} is unresolved — no tests yet.")
            else:
                errors.append(f"{crit.req_id} has no linked test cases.")
            uncovered.append(crit.req_id)

    # Flag assumptions
    for crit in criteria:
        if crit.status == "assumed":
            warn.append(f"{crit.req_id} is assumed: {crit.assumption_note}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warn,
        uncovered_requirements=uncovered,
        invalid_links=invalid_links,
    )
