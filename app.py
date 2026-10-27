"""
Requirement-Based Test Generation Agent — Main Streamlit Application
Groq only. Dark mode only.
"""

import os
import warnings
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*non-supported Python version.*")
warnings.filterwarnings("ignore", message=".*Convert_system_message_to_human.*")

load_dotenv(override=False)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TestGen Agent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ─────────────────────────────────────────────────────────────
from src.models import AgentState
from src.agent import run_full_workflow, continue_after_clarification, skip_clarification
from src.storage import (
    save_project, load_project, list_projects, delete_project,
    export_to_json, export_to_csv, parse_document,
)
from src.ui_components import (
    render_criterion_card, render_test_card, render_validation_panel,
    render_metrics, render_clarification_form,
    CATEGORY_COLORS, PRIORITY_COLORS,
)
from src.llm_tools import GROQ_MODELS

# ══════════════════════════════════════════════════════════════════════════════
# Session init (must run before any st.* calls that read session_state)
# ══════════════════════════════════════════════════════════════════════════════
def _init_session():
    defaults = {
        "state":              AgentState(),
        "project_id":         None,
        "page":               "new",
        "prog_log":           [],
        "_project_name_val":  "",  # backing store for the project name text input
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

def _inject_theme_css():
    """Inject dark mode CSS. Dark mode is the only supported mode."""
    DARK = True  # always dark — light mode removed

    # Palette — dark mode only
    if DARK:
        BG       = "#0d1117"
        BG2      = "#161b22"
        BG_CARD  = "#21262d"
        BORDER   = "#30363d"
        BORDER2  = "#484f58"
        TXT      = "#e6edf3"
        TXT2     = "#8b949e"
        TXT3     = "#656d76"
        SIDEBAR  = "#010409"
        SB_BDR   = "#21262d"
        HERO_GR  = "linear-gradient(135deg,#0d1117 0%,#0c2461 60%,#0d1117 100%)"
        STEP_BG  = "#161b22"
        TIPS_BG  = "rgba(37,99,235,0.1)"
        TIPS_BDR = "rgba(37,99,235,0.3)"

    SB_TXT   = "#e6edf3"
    ACCENT   = "#2563eb"

    st.markdown(f"""
<style>
/* ── Base & transition ────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
  font-family: 'Inter','Segoe UI',system-ui,sans-serif;
  transition: background 0.25s ease, color 0.25s ease;
}}
.stApp {{
  background: {BG} !important;
  transition: background 0.25s ease;
}}
.block-container {{
  padding-top: 1.4rem !important;
  background: {BG} !important;
  transition: background 0.25s ease;
}}
/* Force Streamlit's root section to take our background */
section[data-testid="stMain"] > div,
section.main > div {{
  background: {BG} !important;
}}

/* ── Sidebar (always dark regardless of mode) ─────────────────────────────── */
[data-testid="stSidebar"] {{
  background: {SIDEBAR} !important;
  border-right: 1px solid {SB_BDR} !important;
  transition: background 0.25s ease;
}}
[data-testid="stSidebar"] * {{ color: {SB_TXT} !important; }}
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] input {{
  background: #21262d !important;
  border-color: #30363d !important;
  color: {SB_TXT} !important;
  border-radius: 7px !important;
}}
[data-testid="stSidebar"] .stButton > button {{
  background: #21262d !important;
  border: 1px solid #30363d !important;
  color: {SB_TXT} !important;
  border-radius: 8px;
  font-weight: 500;
  width: 100%;
  text-align: left;
  padding: 9px 14px;
  transition: background 0.15s, border-color 0.15s, transform 0.12s;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
  background: #2d333b !important;
  border-color: {ACCENT} !important;
  transform: translateX(3px);
}}
[data-testid="stSidebar"] .theme-btn > button {{
  background: transparent !important;
  border: 1px solid #30363d !important;
  border-radius: 20px !important;
  font-size: 0.78rem !important;
  padding: 5px 14px !important;
  color: {SB_TXT} !important;
  transition: background 0.15s, border-color 0.15s;
}}
[data-testid="stSidebar"] .theme-btn > button:hover {{
  background: #2d333b !important;
  border-color: #58a6ff !important;
}}

/* ── Hero (always dark gradient) ─────────────────────────────────────────── */
.hero {{
  background: {HERO_GR};
  border-radius: 14px;
  padding: 26px 32px 22px;
  margin-bottom: 22px;
  border: 1px solid #1e3a5f;
}}
.hero h1 {{ color:#f0f6fc; margin:0 0 6px; font-size:1.85rem; font-weight:800; }}
.hero p  {{ color:#94a3b8; margin:0; font-size:0.95rem; }}

/* ── Section heading ──────────────────────────────────────────────────────── */
.sec-hdr {{
  font-size:0.97rem; font-weight:700; color:{TXT};
  border-bottom:2px solid {BORDER}; padding-bottom:6px; margin:20px 0 12px;
  transition: color 0.25s, border-color 0.25s;
}}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card {{
  background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px;
  padding:14px 18px; margin-bottom:10px;
  transition: box-shadow 0.18s, transform 0.15s, background 0.25s, border-color 0.25s;
}}
.card:hover {{ box-shadow:0 4px 18px rgba(0,0,0,0.12); transform:translateY(-1px); }}
.card-title  {{ font-weight:700; font-size:0.96rem; color:{TXT}; transition:color 0.25s; }}
.card-body   {{ margin:6px 0 4px; color:{TXT2}; font-size:0.87rem; line-height:1.6; transition:color 0.25s; }}
.card-source {{ margin:2px 0; font-size:0.75rem; color:{TXT3}; font-style:italic; transition:color 0.25s; }}

/* ── Step tracker ─────────────────────────────────────────────────────────── */
.step-item {{
  flex:1; display:flex; flex-direction:column; align-items:center;
  justify-content:center; padding:11px 5px; gap:3px;
  background:{STEP_BG}; border-right:1px solid {BORDER};
  position:relative; transition:background 0.2s;
}}
.step-item:last-child {{ border-right:none; }}
.step-icon  {{ font-size:1.2rem; }}
.step-label {{ font-size:0.69rem; font-weight:700; text-transform:uppercase;
               letter-spacing:0.05em; color:{TXT3}; transition:color 0.25s; }}
.step-desc  {{ font-size:0.63rem; color:{TXT3}; text-align:center;
               line-height:1.3; max-width:88px; transition:color 0.25s; }}
.step-done  {{ background:rgba(22,163,74,0.1); }}
.step-done  .step-label, .step-done  .step-desc {{ color:#16a34a; }}
.step-active {{ background:rgba(37,99,235,0.1); }}
.step-active .step-label, .step-active .step-desc {{ color:#2563eb; }}
.step-active::after {{
  content:''; position:absolute; bottom:0; left:50%; transform:translateX(-50%);
  width:55%; height:3px; background:#3b82f6; border-radius:3px 3px 0 0;
}}
.step-pending .step-label, .step-pending .step-desc {{ opacity:0.4; }}

/* ── Progress log ─────────────────────────────────────────────────────────── */
.progress-box {{
  background:{BG2}; border:1px solid {BORDER}; border-radius:10px;
  padding:14px 18px; margin:14px 0;
  font-family:'Menlo','Consolas',monospace; font-size:0.81rem; color:{TXT2};
  min-height:72px; transition:background 0.25s, color 0.25s;
}}
.plog-done   {{ color:#16a34a; }}
.plog-active {{ color:#2563eb; font-weight:600; }}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {{
  background:linear-gradient(135deg,#2563eb,#1d4ed8) !important;
  color:#fff !important; border:none !important; border-radius:9px;
  font-weight:700; font-size:0.94rem;
  transition:transform 0.12s, box-shadow 0.18s;
  box-shadow:0 2px 8px rgba(37,99,235,0.3);
}}
.stButton > button[kind="primary"]:hover {{
  transform:translateY(-1px); box-shadow:0 4px 16px rgba(37,99,235,0.4);
}}
.stButton > button[kind="primary"]:active {{ transform:translateY(0); }}
.save-btn > button {{
  background:linear-gradient(135deg,#059669,#047857) !important;
  color:#fff !important; border:none !important; border-radius:9px;
  font-weight:700; font-size:0.87rem;
  transition:transform 0.12s, box-shadow 0.18s;
  box-shadow:0 2px 8px rgba(5,150,105,0.3);
}}
.save-btn > button:hover {{
  transform:translateY(-1px); box-shadow:0 4px 14px rgba(5,150,105,0.4);
}}

/* ── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
  border:2px dashed {BORDER} !important; border-radius:10px !important;
  background:{BG2} !important; transition:border-color 0.2s, background 0.25s;
}}
[data-testid="stFileUploader"]:hover {{ border-color:{ACCENT} !important; }}
[data-testid="stFileUploader"] label {{ color:{TXT} !important; }}
[data-testid="stFileUploader"] p, [data-testid="stFileUploader"] span {{ color:{TXT2} !important; }}

/* ── Inputs & textarea ────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {{
  border-radius:9px !important; border-color:{BORDER} !important;
  background:{BG_CARD} !important; color:{TXT} !important; font-size:0.91rem;
  transition: border-color 0.15s, box-shadow 0.15s, background 0.25s;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
  border-color:{ACCENT} !important;
  box-shadow:0 0 0 3px rgba(37,99,235,0.15) !important;
}}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {{
  color:{TXT3} !important; opacity:0.75;
}}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  gap:4px; border-bottom-color:{BORDER} !important;
  background:{BG} !important;
}}
.stTabs [data-baseweb="tab"] {{
  border-radius:8px 8px 0 0; font-weight:600; font-size:0.86rem;
  color:{TXT2} !important; background:{BG} !important;
  transition:background 0.15s, color 0.15s;
}}
.stTabs [data-baseweb="tab"]:hover {{
  background:{BG2} !important; color:{TXT} !important;
}}
.stTabs [aria-selected="true"] {{
  color:{TXT} !important; background:{BG_CARD} !important;
}}
/* Tab panel content area */
.stTabs [data-baseweb="tab-panel"] {{
  background:{BG} !important;
}}

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {{
  background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px;
  padding:13px; transition:transform 0.15s, background 0.25s, border-color 0.25s;
}}
[data-testid="metric-container"]:hover {{ transform:translateY(-2px); }}
[data-testid="metric-container"] label {{ color:{TXT2} !important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color:{TXT} !important; }}

/* ── Expanders ────────────────────────────────────────────────────────────── */
details {{
  border-radius:10px !important; border-color:{BORDER} !important;
  background:{BG_CARD} !important;
  transition: background 0.25s, border-color 0.25s;
}}
details summary {{
  color:{TXT} !important;
  transition: color 0.25s;
}}
details > div {{
  background:{BG_CARD} !important; color:{TXT} !important;
}}

/* ── Selectbox & multiselect (main area) ─────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] {{
  background:{BG_CARD} !important; border-color:{BORDER} !important;
  transition: background 0.25s, border-color 0.25s;
}}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div {{
  color:{TXT} !important;
}}
.stMultiSelect [data-baseweb="select"] > div {{
  background:{BG_CARD} !important; border-color:{BORDER} !important;
}}
.stMultiSelect [data-baseweb="tag"] {{
  background:{BG2} !important; border:1px solid {BORDER} !important;
  transition: background 0.15s;
}}
.stMultiSelect [data-baseweb="tag"] span {{ color:{TXT} !important; }}
.stMultiSelect [data-baseweb="tag"]:hover {{ background:{BORDER} !important; }}

/* Dropdown list popup */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {{
  background:{BG_CARD} !important; border-color:{BORDER} !important;
}}
[data-baseweb="menu"] li {{ color:{TXT} !important; }}
[data-baseweb="menu"] li:hover {{ background:{BG2} !important; }}

/* ── Info / success / warning / error boxes ──────────────────────────────── */
[data-testid="stAlert"] {{
  background:{BG_CARD} !important;
  transition: background 0.25s;
}}
div[data-testid="stAlert"] p {{ color:{TXT} !important; }}

/* Streamlit native info/success/warning boxes */
.stAlert > div {{ color:{TXT} !important; }}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
  background:{BG_CARD} !important;
  transition: background 0.25s;
}}

/* ── All text in main area ────────────────────────────────────────────────── */
.stApp p, .stApp li, .stApp span:not([data-baseweb]),
.stApp div:not([class*="sidebar"]) {{ color:{TXT}; transition:color 0.25s; }}
label {{ color:{TXT2} !important; transition:color 0.25s; }}
h1,h2,h3,h4,h5,h6 {{ color:{TXT} !important; transition:color 0.25s; }}
.stMarkdown p, .stMarkdown li {{ color:{TXT} !important; }}
.stMarkdown strong {{ color:{TXT} !important; font-weight:700; }}
.stMarkdown em {{ color:{TXT2} !important; }}
[data-testid="stMarkdownContainer"] p {{ color:{TXT} !important; }}
[data-testid="stMarkdownContainer"] li {{ color:{TXT} !important; }}
[data-testid="stMarkdownContainer"] strong {{ color:{TXT} !important; }}
[data-testid="stCaptionContainer"] p {{ color:{TXT3} !important; }}
code {{ background:{BG2} !important; color:{TXT} !important;
        border:1px solid {BORDER}; border-radius:4px; }}
pre  {{ background:{BG2} !important; color:{TXT} !important;
        border:1px solid {BORDER}; border-radius:8px; }}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar       {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{BG2}; border-radius:3px; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px;
                              transition: background 0.2s; }}
::-webkit-scrollbar-thumb:hover {{ background:{BORDER2}; }}
</style>
""", unsafe_allow_html=True)

    # Export palette to module-level so page functions can use them
    import builtins
    g = globals()
    for _var in ["DARK","BG","BG2","BG_CARD","BORDER","BORDER2","TXT","TXT2","TXT3",
                 "SIDEBAR","SB_BDR","SB_TXT","HERO_GR","STEP_BG","TIPS_BG","TIPS_BDR","ACCENT"]:
        g[_var] = locals()[_var]


# ══════════════════════════════════════════════════════════════════════════════
# Workflow steps
# ══════════════════════════════════════════════════════════════════════════════
WORKFLOW_STEPS = [
    ("📥", "Ingest",   "idle",                   "Accept & normalise requirements"),
    ("🔍", "Analyze",  "analyzing",              "Detect ambiguities & gaps"),
    ("🤔", "Clarify",  "awaiting_clarification", "Ask targeted questions"),
    ("📋", "Criteria", "extracting_criteria",    "Extract REQ-IDs & criteria"),
    ("🧪", "Generate", "generating_tests",       "Create structured test cases"),
    ("✅", "Validate", "validating",             "Check traceability links"),
    ("📊", "Report",   "complete",               "Review results & export"),
]
ALL_STATUSES = [s[2] for s in WORKFLOW_STEPS]

def _step_index(status: str) -> int:
    try:    return ALL_STATUSES.index(status)
    except: return 0


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _api_key_set() -> bool:
    """Return True only if a real (non-placeholder) Groq API key is set."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return False
    bad = ("your-key", "...", "gsk_...")
    return not any(b in key for b in bad)


def _persist_key(env_var: str, value: str):
    """Write any env var into .env so it survives Streamlit reruns."""
    path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        lines = open(path).readlines()
        done  = False
        for i,l in enumerate(lines):
            if l.startswith(f"{env_var}="):
                lines[i] = f"{env_var}={value}\n"; done=True; break
        if not done: lines.append(f"{env_var}={value}\n")
        open(path,"w").writelines(lines)
    except: pass


def _provider_label() -> str:
    return f"Groq · {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}"


def _reset():
    st.session_state.state      = AgentState()
    st.session_state.project_id = None
    st.session_state.page       = "new"
    st.session_state.prog_log   = []
    st.session_state["_project_name_val"] = ""




# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:

        # ── Brand ──────────────────────────────────────────────────────────
        st.markdown(
            '<div style="padding:16px 4px 6px;">'
            '<div style="font-size:1.3rem;font-weight:800;color:#f0f6fc;">'
            '🧪 TestGen Agent</div>'
            '<div style="font-size:0.7rem;color:#656d76;margin-top:1px;">'
            'Requirement → Traceable Tests</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(f'<hr style="border-color:{SB_BDR};margin:8px 0;">', unsafe_allow_html=True)

        # ── Navigation ──────────────────────────────────────────────────────
        if st.button("➕  New Analysis", use_container_width=True):
            _reset(); st.rerun()
        if st.button("📁  Saved Projects", use_container_width=True):
            st.session_state.page = "projects"; st.rerun()

        st.markdown(f'<hr style="border-color:{SB_BDR};margin:10px 0;">', unsafe_allow_html=True)

        # ── LLM config — Groq only ───────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;color:#656d76;padding:2px 0 6px;">⚙️ Groq Model</div>',
            unsafe_allow_html=True,
        )
        g_model = st.selectbox(
            "gm", list(GROQ_MODELS.keys()),
            format_func=lambda k: GROQ_MODELS[k], index=0,
            label_visibility="collapsed",
        )
        os.environ["GROQ_MODEL"] = g_model
        _persist_key("GROQ_MODEL", g_model)
        st.markdown(
            '<div style="font-size:0.7rem;color:#475569;margin-top:6px;">'
            '🔗 <a href="https://console.groq.com/keys" target="_blank" '
            'style="color:#58a6ff;">Get free Groq API key →</a></div>',
            unsafe_allow_html=True,
        )

        # Key status
        ok = _api_key_set()
        st.markdown(
            f'<div style="font-size:0.75rem;color:#8b949e;margin-top:9px;">'
            f'{"🟢 API key configured" if ok else "🔴 API key not set"}</div>',
            unsafe_allow_html=True,
        )
        if not ok:
            entered = st.text_input(
                "key_input", type="password", placeholder="gsk_...",
                label_visibility="collapsed",
            )
            if entered.strip():
                os.environ["GROQ_API_KEY"] = entered.strip()
                _persist_key("GROQ_API_KEY", entered.strip())
                st.rerun()

        st.markdown(f'<hr style="border-color:{SB_BDR};margin:12px 0;">', unsafe_allow_html=True)

        # ── Workflow tracker ─────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;color:#656d76;padding:2px 0 7px;">📍 Workflow</div>',
            unsafe_allow_html=True,
        )
        state: AgentState = st.session_state.state
        cur = _step_index(state.status)
        for i,(icon,label,_s,desc) in enumerate(WORKFLOW_STEPS):
            if   i < cur:  dot,lc,dc,bg = "✅","#3fb950","#3fb950","rgba(22,163,74,0.07)"
            elif i == cur: dot,lc,dc,bg = "▶️","#58a6ff","#58a6ff","rgba(88,166,255,0.1)"
            else:          dot,lc,dc,bg = "⬜","#484f58","#484f58","transparent"
            st.markdown(
                f'<div style="display:flex;align-items:flex-start;gap:7px;'
                f'padding:5px 7px;border-radius:7px;background:{bg};margin-bottom:2px;">'
                f'<span style="font-size:0.82rem;min-width:18px;">{dot}</span>'
                f'<div><div style="font-size:0.8rem;font-weight:700;color:{lc};">{icon} {label}</div>'
                f'<div style="font-size:0.66rem;color:{dc};opacity:0.85;line-height:1.3;">{desc}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        if state.status == "complete" and state.project_name:
            st.markdown(f'<hr style="border-color:{SB_BDR};margin:10px 0;">', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(22,163,74,0.1);border:1px solid #3fb950;'
                f'border-radius:8px;padding:8px 11px;font-size:0.77rem;color:#3fb950;">'
                f'💾 <strong>{state.project_name}</strong></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Progress tracker (main body)
# ══════════════════════════════════════════════════════════════════════════════
def render_progress_tracker(current_status: str, log: list):
    cur  = _step_index(current_status)
    cols = st.columns(len(WORKFLOW_STEPS))
    for i,(icon,label,_s,desc) in enumerate(WORKFLOW_STEPS):
        cls = "step-done" if i<cur else ("step-active" if i==cur else "step-pending")
        with cols[i]:
            st.markdown(
                f'<div class="step-item {cls}">'
                f'<span class="step-icon">{icon}</span>'
                f'<span class="step-label">{label}</span>'
                f'<span class="step-desc">{desc}</span></div>',
                unsafe_allow_html=True,
            )
    if log:
        done_html   = "".join(f'<div class="plog-done">✔ {l}</div>' for l in log[:-1])
        active_html = f'<div class="plog-active">⟳ {log[-1]}</div>'
        st.markdown(
            f'<div class="progress-box">{done_html}{active_html}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Page: New Analysis
# ══════════════════════════════════════════════════════════════════════════════
def page_new():
    state: AgentState = st.session_state.state

    st.markdown("""
    <div class="hero">
      <h1>🧪 Requirement-Based Test Generation Agent</h1>
      <p>Paste requirements or upload a document — the agent clarifies ambiguities,
         extracts acceptance criteria, and generates structured, traceable test cases.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">📥 Project Requirements</div>', unsafe_allow_html=True)

    col_main, col_upload = st.columns([3, 2], gap="large")

    with col_main:
        # Use a separate backing key (_project_name_val) that is NOT bound to any widget.
        # This lets us update it freely from the upload section without triggering
        # the "cannot modify after widget instantiation" error.
        if "_project_name_val" not in st.session_state:
            st.session_state["_project_name_val"] = state.project_name or ""

        project_name = st.text_input(
            "Project / Feature Name",
            value=st.session_state["_project_name_val"],
            placeholder="e.g. User Authentication, Cart Discount System, Password Reset…",
        )
        st.markdown(
            f'<div style="font-size:0.84rem;font-weight:600;color:{TXT2};margin:10px 0 4px;">'
            f'Requirements Description</div>', unsafe_allow_html=True,
        )
        requirements_text = st.text_area(
            "req_input", value=state.raw_requirements, height=300,
            label_visibility="collapsed",
            placeholder=(
                "Paste your full feature requirements here.\n\n"
                "Example:\n"
                "1. A registered customer receives a 10% discount when cart total > $100.\n"
                "2. The discount is applied before the checkout confirmation screen.\n"
                "3. Guest users are not eligible for automatic discounts.\n"
                "4. Only one discount code can be applied per order.\n\n"
                "Tip: Include numeric thresholds, user roles and error conditions."
            ),
        )

    with col_upload:
        st.markdown(
            f'<div style="font-size:0.84rem;font-weight:600;color:{TXT2};margin:0 0 7px;">'
            f'Or Upload a Document</div>', unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload requirements document",
            type=["txt", "md", "docx", "pdf"],
            label_visibility="collapsed",
            help="Drag & drop or click to browse. Supported: .txt .md .docx .pdf",
        )
        st.markdown(
            f'<div style="font-size:0.71rem;color:{TXT3};margin-top:3px;">'
            f'📄 .txt · .md · .docx · .pdf &nbsp;|&nbsp; Max: 10 MB</div>',
            unsafe_allow_html=True,
        )

        if uploaded:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("File exceeds 10 MB limit.")
            else:
                extracted = parse_document(uploaded.read(), uploaded.name)
                requirements_text = extracted
                st.success(f"✅ Extracted **{len(extracted):,}** chars from `{uploaded.name}`")

                # Manual project name input — appears after upload
                # Uses its own key (doc_project_name), completely separate from
                # the main project name field, so no session state conflict.
                doc_project = st.text_input(
                    "📁 Project name for this document",
                    key="doc_project_name",
                    placeholder="Enter a project name…",
                )
                if doc_project.strip() and doc_project.strip() != project_name:
                    # Store in the backing variable and rerun so main field picks it up
                    st.session_state["_project_name_val"] = doc_project.strip()
                    st.rerun()

                with st.expander("Preview extracted text", expanded=False):
                    st.text(extracted[:1500] + ("…" if len(extracted) > 1500 else ""))

        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{TIPS_BG};border:1px solid {TIPS_BDR};'
            f'border-radius:10px;padding:14px 16px;font-size:0.81rem;color:{TXT2};line-height:1.75;">'
            f'<strong style="color:{TXT};">💡 Tips for better results</strong><br>'
            f'• Include numeric thresholds (e.g. "more than $100")<br>'
            f'• Mention user roles (admin, guest, registered user)<br>'
            f'• Describe error conditions and edge cases<br>'
            f'• Ambiguous requirements → agent asks clarifying questions'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    btn_col, info_col = st.columns([2, 3])
    with btn_col:
        run = st.button(
            "🚀  Analyze & Generate Tests", type="primary",
            use_container_width=True,
            disabled=not (requirements_text or "").strip(),
        )
    with info_col:
        ok = _api_key_set()
        color = "#3fb950" if ok else "#f85149"
        dot   = "🟢" if ok else "🔴"
        msg   = f"Using <strong>{_provider_label()}</strong>" if ok else "API key missing — enter it in the sidebar"
        st.markdown(
            f'<div style="font-size:0.8rem;color:{color};padding-top:10px;">{dot} {msg}</div>',
            unsafe_allow_html=True,
        )

    if run and (requirements_text or "").strip():
        if not _api_key_set():
            st.error("⚠️ Please enter your API key in the sidebar before running.")
            return

        st.session_state.prog_log = []
        state.raw_requirements    = requirements_text.strip()
        state.project_name        = (project_name or "").strip() or "Untitled Project"
        state.status              = "analyzing"

        ph       = st.empty()
        prog_log = []

        def update_progress(msg: str):
            prog_log.append(msg)
            with ph.container():
                render_progress_tracker(state.status, prog_log)

        try:
            update_progress("Analysing requirements for ambiguities and unclear statements…")
            state = run_full_workflow(state, progress_cb=update_progress)
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            state.status = "error"; state.error_message = str(e)
            st.session_state.state = state; return

        st.session_state.state    = state
        st.session_state.prog_log = prog_log

        if state.status == "awaiting_clarification":
            st.session_state.page = "clarify"
        else:
            pid = save_project(state)
            st.session_state.project_id = pid
            st.session_state.page       = "results"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Clarification
# ══════════════════════════════════════════════════════════════════════════════
def page_clarify():
    state: AgentState = st.session_state.state
    st.markdown("""
    <div class="hero">
      <h1>🤔 Clarification Needed</h1>
      <p>Some requirements are ambiguous. Answer these questions to improve test accuracy.</p>
    </div>""", unsafe_allow_html=True)

    if state.ambiguity_analysis:
        st.markdown(
            f'<div style="background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.35);'
            f'border-radius:10px;padding:13px 17px;margin-bottom:14px;font-size:0.87rem;color:{TXT2};">'
            f'<strong>🔍 Agent summary:</strong> {state.ambiguity_analysis.summary}</div>',
            unsafe_allow_html=True,
        )
        if state.ambiguity_analysis.issues:
            with st.expander("View identified issues"):
                for issue in state.ambiguity_analysis.issues:
                    st.markdown(f"- ⚠️ {issue}")

    answers, submitted, skip_all = render_clarification_form(state.clarification_questions)

    if submitted or skip_all:
        if not _api_key_set():
            st.error("⚠️ API key missing — enter it in the sidebar.")
            return
        ph       = st.empty()
        prog_log = []
        def update_progress(msg: str):
            prog_log.append(msg)
            with ph.container():
                render_progress_tracker(state.status, prog_log)
        try:
            if skip_all:
                state = skip_clarification(state, progress_cb=update_progress)
            else:
                filtered = {k:v for k,v in answers.items() if v and v.strip()}
                state = continue_after_clarification(state, filtered, progress_cb=update_progress)
        except Exception as e:
            st.error(f"❌ Error: {e}"); return

        st.session_state.state = state
        pid = save_project(state, st.session_state.project_id)
        st.session_state.project_id = pid
        st.session_state.page       = "results"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Results
# ══════════════════════════════════════════════════════════════════════════════
def page_results():
    state: AgentState = st.session_state.state
    if state.status not in ("complete","validating"):
        st.warning("No results yet. Start a new analysis."); return

    # Hero + save button
    hc, sc = st.columns([5,1])
    with hc:
        st.markdown(
            f'<div class="hero" style="margin-bottom:16px;">'
            f'<h1>📊 {state.project_name}</h1>'
            f'<p>Generated <strong>{len(state.test_cases)}</strong> test cases across '
            f'<strong>{len(state.acceptance_criteria)}</strong> acceptance criteria.</p>'
            f'</div>', unsafe_allow_html=True,
        )
    with sc:
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="save-btn">', unsafe_allow_html=True)
        if st.button("💾 Save Project", use_container_width=True):
            pid = save_project(state, st.session_state.project_id)
            st.session_state.project_id = pid
            st.toast("✅ Project saved!", icon="💾")
        st.markdown('</div>', unsafe_allow_html=True)

    if state.validation_result:
        render_metrics(state.acceptance_criteria, state.test_cases, state.validation_result)

    st.markdown("---")

    t_ov, t_cr, t_tc, t_mx, t_ex = st.tabs([
        "📋 Overview", "📌 Acceptance Criteria",
        "🧪 Test Cases", "🔗 Traceability Matrix", "📤 Export",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with t_ov:
        ca, cb = st.columns(2, gap="large")
        with ca:
            if state.validation_result:
                render_validation_panel(state.validation_result)
            if state.assumptions:
                st.markdown('<div class="sec-hdr">📌 Recorded Assumptions</div>', unsafe_allow_html=True)
                for a in state.assumptions: st.warning(a)
        with cb:
            st.markdown('<div class="sec-hdr">📊 Test Case Categories</div>', unsafe_allow_html=True)
            by_cat: dict = {}
            for tc in state.test_cases: by_cat[tc.category] = by_cat.get(tc.category,0)+1
            tot = len(state.test_cases) or 1
            for cat,cnt in sorted(by_cat.items(), key=lambda x:-x[1]):
                c = CATEGORY_COLORS.get(cat,"#656d76"); w = int(100*cnt/tot)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:7px;">'
                    f'<span style="width:115px;font-size:0.81rem;color:{TXT2};">{cat}</span>'
                    f'<div style="flex:1;background:{BORDER};border-radius:4px;height:15px;">'
                    f'<div style="width:{w}%;background:{c};height:100%;border-radius:4px;"></div></div>'
                    f'<span style="width:20px;text-align:right;font-weight:700;font-size:0.81rem;color:{c};">{cnt}</span>'
                    f'</div>', unsafe_allow_html=True,
                )
            st.markdown('<div class="sec-hdr">🎯 Priority Distribution</div>', unsafe_allow_html=True)
            by_pri: dict = {}
            for tc in state.test_cases: by_pri[tc.priority] = by_pri.get(tc.priority,0)+1
            for pri in ["high","medium","low"]:
                if pri not in by_pri: continue
                cnt = by_pri[pri]; c = PRIORITY_COLORS.get(pri,"#656d76"); w = int(100*cnt/tot)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:7px;">'
                    f'<span style="width:55px;font-size:0.81rem;color:{TXT2};">{pri}</span>'
                    f'<div style="flex:1;background:{BORDER};border-radius:4px;height:15px;">'
                    f'<div style="width:{w}%;background:{c};height:100%;border-radius:4px;"></div></div>'
                    f'<span style="width:20px;text-align:right;font-weight:700;font-size:0.81rem;color:{c};">{cnt}</span>'
                    f'</div>', unsafe_allow_html=True,
                )
        answered = [q for q in state.clarification_questions if q.answered]
        if answered:
            st.markdown('<div class="sec-hdr">💬 Clarification Q&A</div>', unsafe_allow_html=True)
            for q in answered:
                short = (q.question[:90]+"…") if len(q.question)>90 else q.question
                with st.expander(f"Q: {short}"):
                    st.markdown(f"**Question:** {q.question}")
                    st.markdown(f"**Why it mattered:** {q.reason}")
                    st.success(f"**Your answer:** {q.answer}")

    # ── Criteria ──────────────────────────────────────────────────────────────
    with t_cr:
        st.markdown('<div class="sec-hdr">📌 Acceptance Criteria</div>', unsafe_allow_html=True)
        sf = st.multiselect("Filter by status", ["confirmed","assumed","unresolved"],
                            default=["confirmed","assumed","unresolved"])
        shown = [c for c in state.acceptance_criteria if c.status in sf]
        st.caption(f"Showing {len(shown)} of {len(state.acceptance_criteria)} criteria")
        for i,crit in enumerate(shown): render_criterion_card(crit, i)

    # ── Test cases ────────────────────────────────────────────────────────────
    with t_tc:
        st.markdown('<div class="sec-hdr">🧪 Generated Test Cases</div>', unsafe_allow_html=True)
        f1,f2,f3 = st.columns(3)
        with f1: cat_f = st.multiselect("Category", list(CATEGORY_COLORS.keys()), default=list(CATEGORY_COLORS.keys()))
        with f2: pri_f = st.multiselect("Priority", ["high","medium","low"], default=["high","medium","low"])
        with f3: req_f = st.multiselect("Requirement", [c.req_id for c in state.acceptance_criteria],
                                         default=[c.req_id for c in state.acceptance_criteria])
        shown_tc = [tc for tc in state.test_cases
                    if tc.category in cat_f and tc.priority in pri_f
                    and any(r in req_f for r in tc.requirement_ids)]
        st.caption(f"Showing {len(shown_tc)} of {len(state.test_cases)} test cases")
        for i,tc in enumerate(shown_tc): render_test_card(tc, i)

    # ── Traceability matrix ───────────────────────────────────────────────────
    with t_mx:
        st.markdown('<div class="sec-hdr">🔗 Requirement–Test Traceability Matrix</div>', unsafe_allow_html=True)
        st.caption("✅ = test case covers the requirement")
        if state.acceptance_criteria and state.test_cases:
            import pandas as pd
            req_ids = [c.req_id for c in state.acceptance_criteria]
            mx      = {tc.test_id: {r:("✅" if r in tc.requirement_ids else "") for r in req_ids}
                       for tc in state.test_cases}
            df = pd.DataFrame(mx).T; df.index.name = "Test ↓ / Req →"
            def _g(v): return "background-color:#1a4731;color:#3fb950;" if v=="✅" else ""
            st.dataframe(df.style.applymap(_g), use_container_width=True)
            cov = [{"ID":cr.req_id,"Title":cr.title,
                    "Tests"  :len([t for t in state.test_cases if cr.req_id in t.requirement_ids]),
                    "Status" :cr.status,
                    "Covered":"✅" if any(cr.req_id in t.requirement_ids for t in state.test_cases) else "❌"}
                   for cr in state.acceptance_criteria]
            st.markdown("**Coverage summary:**")
            st.dataframe(pd.DataFrame(cov), use_container_width=True, hide_index=True)
        else:
            st.info("No data yet.")

    # ── Export ────────────────────────────────────────────────────────────────
    with t_ex:
        st.markdown('<div class="sec-hdr">📤 Export Results</div>', unsafe_allow_html=True)
        csv_data  = export_to_csv(state)
        json_data = export_to_json(state)
        safe      = state.project_name.replace(" ","_")
        ec1, ec2  = st.columns(2)
        with ec1:
            st.download_button("⬇️  Download CSV",  data=csv_data,  file_name=f"{safe}_tests.csv",
                               mime="text/csv", use_container_width=True)
            st.caption("Excel / Google Sheets compatible")
        with ec2:
            st.download_button("⬇️  Download JSON", data=json_data, file_name=f"{safe}_tests.json",
                               mime="application/json", use_container_width=True)
            st.caption("Full schema with traceability links")
        with st.expander("Preview JSON"):
            st.code(json_data[:3000]+("…" if len(json_data)>3000 else ""), language="json")
        with st.expander("Preview CSV"):
            st.code(csv_data[:2000]+ ("…" if len(csv_data) >2000 else ""), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Saved Projects
# ══════════════════════════════════════════════════════════════════════════════
def page_projects():
    st.markdown("""
    <div class="hero">
      <h1>📁 Saved Projects</h1>
      <p>Load a previous analysis to review, re-export, or continue editing its test cases.</p>
    </div>""", unsafe_allow_html=True)

    projects = list_projects()
    if not projects:
        st.info("No saved projects yet. Run an analysis and click **Save Project**.")
        if st.button("➕ Start New Analysis"): _reset(); st.rerun()
        return

    for proj in projects:
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">📂 {proj["name"]}</div>'
            f'<div class="card-source">'
            f'Created: {proj["created_at"][:10]} &nbsp;·&nbsp; '
            f'Updated: {proj["updated_at"][:16].replace("T"," ")}'
            f'</div></div>', unsafe_allow_html=True,
        )
        bc1,bc2,_ = st.columns([1,1,5])
        with bc1:
            if st.button("📂 Load", key=f"l_{proj['id']}", use_container_width=True):
                loaded = load_project(proj["id"])
                if loaded:
                    st.session_state.state      = loaded
                    st.session_state.project_id = proj["id"]
                    st.session_state.page       = "results"
                    st.rerun()
        with bc2:
            if st.button("🗑️ Delete", key=f"d_{proj['id']}", use_container_width=True):
                delete_project(proj["id"]); st.rerun()
        st.markdown(f'<hr style="border-color:{BORDER};margin:6px 0;">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _inject_theme_css()   # must be first — sets palette vars + injects CSS
    render_sidebar()
    state: AgentState = st.session_state.state
    page  = st.session_state.page

    if state.status == "awaiting_clarification" and page != "clarify":
        st.session_state.page = page = "clarify"
    elif state.status == "complete" and page not in ("results","projects"):
        st.session_state.page = page = "results"

    if   page == "projects": page_projects()
    elif page == "clarify":  page_clarify()
    elif page == "results":  page_results()
    else:                    page_new()


if __name__ == "__main__":
    main()
