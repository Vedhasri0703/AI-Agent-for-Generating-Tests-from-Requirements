"""
Pydantic models for the Requirement-Based Test Generation Agent.
All data structures used throughout the agent workflow are defined here.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ---------------------------------------------------------------------------
# Requirement & Criteria Models
# ---------------------------------------------------------------------------

class AcceptanceCriterion(BaseModel):
    """A single structured acceptance criterion extracted from requirements."""
    req_id: str = Field(..., description="Stable ID, e.g. REQ-001")
    title: str = Field(..., description="Short title of the requirement")
    criterion: str = Field(..., description="Explicit, testable acceptance statement")
    source_text: str = Field(..., description="Original text this was derived from")
    status: Literal["confirmed", "unresolved", "assumed"] = Field(
        default="confirmed",
        description="confirmed = user clarified; assumed = agent inferred; unresolved = still unclear"
    )
    assumption_note: Optional[str] = Field(
        default=None,
        description="If status is assumed/unresolved, describe what was assumed or is still unclear"
    )


class ClarificationQuestion(BaseModel):
    """A focused question the agent asks when a requirement is ambiguous."""
    question_id: str = Field(default_factory=lambda: f"Q-{str(uuid.uuid4())[:8].upper()}")
    question: str = Field(..., description="The question text")
    reason: str = Field(..., description="Why this question is needed — which behavior it affects")
    answer: Optional[str] = Field(default=None, description="User's answer")
    answered: bool = Field(default=False)


class AmbiguityAnalysis(BaseModel):
    """Result of analyzing requirements for clarity issues."""
    has_ambiguity: bool = Field(..., description="True if clarification is needed")
    issues: List[str] = Field(default_factory=list, description="List of identified ambiguity issues")
    questions: List[ClarificationQuestion] = Field(
        default_factory=list,
        description="Focused clarification questions"
    )
    summary: str = Field(..., description="Brief summary of what was found")


# ---------------------------------------------------------------------------
# Test Case Models
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    """A fully structured test case linked to one or more requirement IDs."""
    test_id: str = Field(..., description="Unique ID, e.g. TC-001")
    requirement_ids: List[str] = Field(
        ...,
        description="One or more REQ IDs this test covers"
    )
    title: str = Field(..., description="Short descriptive name")
    category: Literal["positive", "negative", "boundary", "validation", "error-handling"] = Field(
        ..., description="Test category"
    )
    preconditions: str = Field(..., description="Required state before the test")
    test_data: str = Field(..., description="Input values or data needed")
    steps: List[str] = Field(..., description="Actions in execution order")
    expected_result: str = Field(..., description="Observable behavior that should occur")
    priority: Literal["high", "medium", "low"] = Field(default="medium")
    review_status: Literal["draft", "reviewed", "needs changes"] = Field(default="draft")
    notes: Optional[str] = Field(default=None)


class ValidationResult(BaseModel):
    """Result of schema and traceability validation."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    uncovered_requirements: List[str] = Field(
        default_factory=list,
        description="REQ IDs with no linked test case"
    )
    invalid_links: List[str] = Field(
        default_factory=list,
        description="test_id:req_id pairs where req_id does not exist"
    )


# ---------------------------------------------------------------------------
# Agent State Model (LangGraph-compatible)
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """Complete state for the test generation workflow."""
    # Input
    raw_requirements: str = Field(default="", description="Original user input")
    project_name: str = Field(default="", description="Optional project/feature name")

    # Analysis
    ambiguity_analysis: Optional[AmbiguityAnalysis] = None

    # Clarification loop
    clarification_questions: List[ClarificationQuestion] = Field(default_factory=list)
    clarification_answers: dict = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)

    # Criteria & tests
    acceptance_criteria: List[AcceptanceCriterion] = Field(default_factory=list)
    test_cases: List[TestCase] = Field(default_factory=list)

    # Validation
    validation_result: Optional[ValidationResult] = None

    # Workflow control
    status: Literal[
        "idle",
        "analyzing",
        "awaiting_clarification",
        "extracting_criteria",
        "generating_tests",
        "validating",
        "complete",
        "error"
    ] = Field(default="idle")

    error_message: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        arbitrary_types_allowed = True
