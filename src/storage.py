"""
Storage layer: SQLite-based project persistence with JSON export/import support.
"""

from __future__ import annotations
import json
import sqlite3
import os
from datetime import datetime
from typing import List, Optional
from src.models import AgentState


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "projects.db")


def _get_conn() -> sqlite3.Connection:
    """Get a database connection, creating the DB and tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            state_json  TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def save_project(state: AgentState, project_id: Optional[int] = None) -> int:
    """Save or update a project. Returns the project ID."""
    conn = _get_conn()
    now = datetime.now().isoformat()
    state_json = state.model_dump_json()
    name = state.project_name or f"Project {now[:10]}"

    if project_id:
        conn.execute(
            "UPDATE projects SET name=?, state_json=?, updated_at=? WHERE id=?",
            (name, state_json, now, project_id)
        )
        conn.commit()
        conn.close()
        return project_id
    else:
        cursor = conn.execute(
            "INSERT INTO projects (name, state_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, state_json, now, now)
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        return new_id


def load_project(project_id: int) -> Optional[AgentState]:
    """Load a project by ID. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT state_json FROM projects WHERE id=?", (project_id,)
    ).fetchone()
    conn.close()
    if row:
        return AgentState.model_validate_json(row["state_json"])
    return None


def list_projects() -> List[dict]:
    """Return a list of all projects (id, name, created_at, updated_at)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name, created_at, updated_at FROM projects ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_project(project_id: int) -> bool:
    """Delete a project. Returns True if it existed."""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def export_to_json(state: AgentState) -> str:
    """Export test cases and criteria to a JSON string."""
    data = {
        "project": state.project_name,
        "exported_at": datetime.now().isoformat(),
        "acceptance_criteria": [c.model_dump() for c in state.acceptance_criteria],
        "test_cases": [t.model_dump() for t in state.test_cases],
    }
    return json.dumps(data, indent=2)


def export_to_csv(state: AgentState) -> str:
    """Export test cases to CSV string."""
    import csv
    import io

    output = io.StringIO()
    fieldnames = [
        "test_id", "requirement_ids", "title", "category",
        "preconditions", "test_data", "steps",
        "expected_result", "priority", "review_status", "notes"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for tc in state.test_cases:
        row = tc.model_dump()
        row["requirement_ids"] = "; ".join(row["requirement_ids"])
        row["steps"] = "\n".join(row["steps"])
        writer.writerow(row)

    return output.getvalue()


def parse_document(file_bytes: bytes, file_name: str) -> str:
    """Parse uploaded document (TXT, DOCX, PDF) and return plain text."""
    ext = os.path.splitext(file_name)[1].lower()

    if ext in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="replace")

    elif ext == ".docx":
        import io
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    elif ext == ".pdf":
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    else:
        return file_bytes.decode("utf-8", errors="replace")
