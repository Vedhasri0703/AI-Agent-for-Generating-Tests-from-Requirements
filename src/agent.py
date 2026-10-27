"""
Agent workflow orchestration.
Implements a state-machine style workflow (LangGraph-compatible structure)
that routes through: ingest → analyze → [clarify loop] → extract → generate → validate.
"""

from __future__ import annotations
from typing import Callable
from src.models import AgentState, ClarificationQuestion
from src.llm_tools import (
    analyze_requirement_ambiguity,
    extract_acceptance_criteria,
    generate_test_cases,
    validate_traceability,
)


# ---------------------------------------------------------------------------
# Node functions — each takes and returns AgentState
# ---------------------------------------------------------------------------

def node_analyze(state: AgentState, progress_cb: Callable[[str], None] | None = None) -> AgentState:
    """Analyze requirements for ambiguity."""
    if progress_cb:
        progress_cb("🔍 Analyzing requirements for ambiguity...")

    answers = {q: a for q, a in state.clarification_answers.items()}
    analysis = analyze_requirement_ambiguity(state.raw_requirements, answers or None)
    state.ambiguity_analysis = analysis
    state.clarification_questions = analysis.questions

    if analysis.has_ambiguity and analysis.questions:
        state.status = "awaiting_clarification"
    else:
        state.status = "extracting_criteria"

    return state


def node_apply_answers(state: AgentState, answers: dict[str, str]) -> AgentState:
    """Apply user clarification answers to state."""
    for q_id, answer in answers.items():
        state.clarification_answers[q_id] = answer

    # Mark questions as answered
    for q in state.clarification_questions:
        if q.question_id in answers:
            q.answer = answers[q.question_id]
            q.answered = True

    # Any unanswered questions become assumptions
    for q in state.clarification_questions:
        if not q.answered:
            assumption = f"Assumed regarding '{q.question}': behavior not specified by user."
            if assumption not in state.assumptions:
                state.assumptions.append(assumption)

    state.status = "extracting_criteria"
    return state


def node_extract_criteria(state: AgentState, progress_cb: Callable[[str], None] | None = None) -> AgentState:
    """Extract acceptance criteria from requirements + answers."""
    if progress_cb:
        progress_cb("📋 Extracting acceptance criteria...")

    answers_dict = {
        q.question: q.answer
        for q in state.clarification_questions
        if q.answered and q.answer
    }

    criteria = extract_acceptance_criteria(
        requirements=state.raw_requirements,
        clarification_answers=answers_dict or None,
        assumptions=state.assumptions or None,
    )
    state.acceptance_criteria = criteria
    state.status = "generating_tests"
    return state


def node_generate_tests(state: AgentState, progress_cb: Callable[[str], None] | None = None) -> AgentState:
    """Generate test cases from acceptance criteria."""
    if progress_cb:
        progress_cb("🧪 Generating test cases...")

    tests = generate_test_cases(
        criteria=state.acceptance_criteria,
        requirements=state.raw_requirements,
    )
    state.test_cases = tests
    state.status = "validating"
    return state


def node_validate(state: AgentState, progress_cb: Callable[[str], None] | None = None) -> AgentState:
    """Validate traceability and schema."""
    if progress_cb:
        progress_cb("✅ Validating traceability...")

    result = validate_traceability(state.acceptance_criteria, state.test_cases)
    state.validation_result = result
    state.status = "complete"
    return state


# ---------------------------------------------------------------------------
# High-level workflow runner
# ---------------------------------------------------------------------------

def run_full_workflow(
    state: AgentState,
    progress_cb: Callable[[str], None] | None = None,
) -> AgentState:
    """
    Run the full workflow from analysis through validation.
    Stops at 'awaiting_clarification' if questions need answers.
    """
    state = node_analyze(state, progress_cb)

    if state.status == "awaiting_clarification":
        return state  # Caller must collect answers and call continue_after_clarification()

    state = node_extract_criteria(state, progress_cb)
    state = node_generate_tests(state, progress_cb)
    state = node_validate(state, progress_cb)
    return state


def continue_after_clarification(
    state: AgentState,
    answers: dict[str, str],
    progress_cb: Callable[[str], None] | None = None,
) -> AgentState:
    """
    Resume workflow after user provides clarification answers.
    """
    state = node_apply_answers(state, answers)
    state = node_extract_criteria(state, progress_cb)
    state = node_generate_tests(state, progress_cb)
    state = node_validate(state, progress_cb)
    return state


def skip_clarification(
    state: AgentState,
    progress_cb: Callable[[str], None] | None = None,
) -> AgentState:
    """
    Skip clarification questions and proceed with assumptions.
    """
    # Mark all unanswered questions as assumptions
    for q in state.clarification_questions:
        if not q.answered:
            assumption = f"Assumed regarding '{q.question}': behavior not specified."
            if assumption not in state.assumptions:
                state.assumptions.append(assumption)

    state.status = "extracting_criteria"
    state = node_extract_criteria(state, progress_cb)
    state = node_generate_tests(state, progress_cb)
    state = node_validate(state, progress_cb)
    return state
