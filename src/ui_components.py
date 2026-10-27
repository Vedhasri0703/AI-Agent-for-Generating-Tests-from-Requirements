"""
Reusable Streamlit UI components — light/dark mode aware.
"""

import streamlit as st
from src.models import AcceptanceCriterion, TestCase, ValidationResult, ClarificationQuestion

# ── Category / Priority / Status colour palette ──────────────────────────────
CATEGORY_COLORS = {
    "positive":      "#16a34a",
    "negative":      "#dc2626",
    "boundary":      "#d97706",
    "validation":    "#2563eb",
    "error-handling":"#9333ea",
}

PRIORITY_COLORS = {
    "high":   "#dc2626",
    "medium": "#d97706",
    "low":    "#16a34a",
}

STATUS_COLORS = {
    "confirmed":    "#16a34a",
    "assumed":      "#d97706",
    "unresolved":   "#dc2626",
    "draft":        "#64748b",
    "reviewed":     "#16a34a",
    "needs changes":"#d97706",
}


# ── Helper: coloured pill badge ───────────────────────────────────────────────
def _badge(label: str, color: str = "#64748b") -> str:
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        f'padding:3px 11px;border-radius:20px;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.03em;">{label}</span>'
    )


# ── Acceptance criteria card ──────────────────────────────────────────────────
def render_criterion_card(crit: AcceptanceCriterion, index: int) -> None:
    sc = STATUS_COLORS.get(crit.status, "#64748b")
    note_html = (
        f'<div style="margin-top:8px;padding:6px 10px;border-radius:6px;'
        f'background:rgba(217,119,6,0.12);font-size:0.78rem;color:#92400e;">'
        f'⚠️ {crit.assumption_note}</div>'
        if crit.assumption_note else ""
    )
    st.markdown(
        f"""
        <div class="card" style="border-left:4px solid {sc};">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
            <span class="card-title">{crit.req_id}: {crit.title}</span>
            {_badge(crit.status, sc)}
          </div>
          <p class="card-body">{crit.criterion}</p>
          <p class="card-source">Source: {crit.source_text[:130]}{'…' if len(crit.source_text)>130 else ''}</p>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Test case card ────────────────────────────────────────────────────────────
def render_test_card(tc: TestCase, index: int) -> None:
    cc  = CATEGORY_COLORS.get(tc.category,    "#64748b")
    pc  = PRIORITY_COLORS.get(tc.priority,    "#64748b")
    sc  = STATUS_COLORS.get(tc.review_status, "#64748b")
    steps_li = "".join(f"<li style='margin:3px 0;'>{s}</li>" for s in tc.steps)
    req_pills = " ".join(
        f'<span style="background:rgba(37,99,235,0.15);color:#1d4ed8;'
        f'padding:2px 9px;border-radius:10px;font-size:0.71rem;font-weight:700;">{r}</span>'
        for r in tc.requirement_ids
    )

    with st.expander(
        f"**{tc.test_id}** · {tc.title}   [{tc.category}]   [{tc.priority}]",
        expanded=False,
    ):
        st.markdown(
            f"""
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
              {_badge(tc.category, cc)}
              {_badge(tc.priority, pc)}
              {_badge(tc.review_status, sc)}
              <span style="font-size:0.8rem;padding-top:4px;">Covers: {req_pills}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Preconditions**")
            st.info(tc.preconditions)
            st.markdown("**Test Data**")
            st.code(tc.test_data, language=None)
        with c2:
            st.markdown("**Steps**")
            st.markdown(
                f"<ol style='margin:0;padding-left:18px;line-height:1.7;'>{steps_li}</ol>",
                unsafe_allow_html=True,
            )
            st.markdown("**Expected Result**")
            st.success(tc.expected_result)
        if tc.notes:
            st.caption(f"📝 {tc.notes}")


# ── Validation panel ──────────────────────────────────────────────────────────
def render_validation_panel(vr: ValidationResult) -> None:
    if vr.is_valid:
        st.success("✅ All traceability checks passed — every test links to a valid requirement.")
    else:
        st.error(f"❌ Validation failed — {len(vr.errors)} error(s) found.")

    if vr.errors:
        with st.expander("🚨 Errors  (must fix)", expanded=True):
            for e in vr.errors:
                st.markdown(f"- 🔴 {e}")

    if vr.warnings:
        with st.expander("⚠️ Warnings  (review recommended)"):
            for w in vr.warnings:
                st.markdown(f"- 🟡 {w}")

    if vr.uncovered_requirements:
        with st.expander("📭 Uncovered Requirements", expanded=True):
            st.caption("These criteria have no linked test cases yet:")
            for r in vr.uncovered_requirements:
                st.markdown(f"- `{r}`")

    if vr.invalid_links:
        with st.expander("🔗 Invalid Links", expanded=True):
            for lnk in vr.invalid_links:
                st.markdown(f"- ❌ `{lnk}`")


# ── Metrics row ───────────────────────────────────────────────────────────────
def render_metrics(criteria, test_cases, vr) -> None:
    total_c  = len(criteria)
    total_t  = len(test_cases)
    covered  = total_c - len(vr.uncovered_requirements) if vr else 0
    valid_t  = total_t - len(vr.invalid_links) if vr else total_t
    cov_pct  = round(100 * covered  / total_c) if total_c else 0
    val_pct  = round(100 * valid_t  / total_t) if total_t else 0
    cats     = len({tc.category for tc in test_cases})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Requirements",         total_c)
    c2.metric("Test Cases",           total_t)
    c3.metric("Req Coverage",         f"{cov_pct}%")
    c4.metric("Test Validity",        f"{val_pct}%")
    c5.metric("Test Categories",      cats)


# ── Clarification form ────────────────────────────────────────────────────────
def render_clarification_form(questions: list[ClarificationQuestion]):
    answers: dict[str, str] = {}
    st.markdown(
        '<p style="font-size:0.9rem;color:var(--text-muted);">'
        'Answer as many as you can — skipped questions will be recorded as assumptions.</p>',
        unsafe_allow_html=True,
    )

    with st.form("clarification_form"):
        for i, q in enumerate(questions, 1):
            st.markdown(
                f"""
                <div style="background:rgba(14,165,233,0.08);border-left:3px solid #0ea5e9;
                            padding:10px 14px;border-radius:8px;margin-bottom:4px;">
                    <strong style="font-size:0.9rem;">Q{i}:</strong>
                    <span style="font-size:0.9rem;"> {q.question}</span><br>
                    <span style="font-size:0.75rem;color:#64748b;font-style:italic;">
                        Why it matters: {q.reason}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            answers[q.question_id] = st.text_area(
                f"answer_q{i}",
                key=f"ans_{q.question_id}",
                placeholder="Your answer (leave blank to skip)…",
                label_visibility="collapsed",
                height=68,
            )

        bc1, bc2 = st.columns([3, 1])
        with bc1:
            submitted = st.form_submit_button(
                "✅  Submit Answers & Generate Tests",
                use_container_width=True, type="primary",
            )
        with bc2:
            skip = st.form_submit_button("⏭️  Skip All", use_container_width=True)

    return answers, submitted, skip
