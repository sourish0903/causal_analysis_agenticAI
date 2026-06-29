"""
Streamlit UI for the Causal AI Agentic Framework.

This app wires the three exported tools from scripts/causal_agentic_ai.py:
- find_optimal_levers
- analyze_whatif_scenario
- analyze_variable_impact

Usage:
- Upload a processed dataframe that contains the columns expected by the tools (Store, Dept, Date, Weekly_Sales, MarkDown*, etc.).
- Choose a tool, fill in parameters, and run.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import pandas as pd
import streamlit as st
from dotenv import dotenv_values
try:
    from huggingface_hub import hf_hub_download  # type: ignore
    try:
        # newer releases expose exceptions here
        from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError  # type: ignore
    except Exception:
        try:
            # older layouts may expose elsewhere
            from huggingface_hub.hf_api import RepositoryNotFoundError, RevisionNotFoundError  # type: ignore
        except Exception:
            # fallback to generic Exception so except (...) still works
            RepositoryNotFoundError = RevisionNotFoundError = Exception
except Exception:
    hf_hub_download = None
    RepositoryNotFoundError = RevisionNotFoundError = Exception

# Ensure scripts/ is on sys.path so imports work when running from repo root
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

import dotenv
# Prefer Streamlit Cloud secrets (safe on deploy). If present, set env var immediately.
try:
    cloud_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    cloud_key = None
if cloud_key:
    os.environ["OPENAI_API_KEY"] = cloud_key

# Load .env into environment (if present) but do NOT overwrite a cloud secret
dotenv.load_dotenv(ROOT.parent / ".env")

# Ensure .env key is cleaned (strip surrounding quotes) and exported to process env
env_path = ROOT.parent / ".env"
if env_path.exists():
    vals = dotenv_values(env_path)
    raw_key = vals.get("OPENAI_API_KEY")
    if raw_key:
        clean_key = raw_key.strip().strip('"\'' )
        os.environ["OPENAI_API_KEY"] = clean_key
        try:
            st.session_state["OPENAI_API_KEY"] = clean_key
        except Exception:
            pass

# Safe logging: show only a masked preview (first 8 chars) in logs
logging.basicConfig(level=logging.INFO)
masked = "<not set>" if not os.environ.get("OPENAI_API_KEY") else (os.environ.get("OPENAI_API_KEY")[:8] + "...")
logging.info("OPENAI_API_KEY visible to process: %s", masked)

from langchain_core.messages import HumanMessage
from causal_agentic_ai import (  # type: ignore
    build_agent,
    init_llm,
    set_global_data,
    set_last_user_question,
)
# Input/output safety guardrails (separate component, see scripts/guardrails.py)
from guardrails import guard_input, guard_output  # type: ignore

# Optional: initialize a lightweight LLM for semantic helpers (no-op if unavailable)
def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "****"
    return f"{k[:4]}...{k[-4:]}"


def ensure_openai_credentials() -> Optional[str]:
    # Priority: env var -> .env -> session_state -> Streamlit secrets -> user prompt
    # Avoid directly indexing st.secrets because Streamlit raises if no secrets file exists.
    key = os.environ.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    # session state may already hold the key from a previous interaction
    try:
        sess_key = st.session_state.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.session_state else None
    except Exception:
        sess_key = None

    if not key and sess_key:
        key = sess_key

    # Try Streamlit secrets safely
    if not key:
        try:
            # st.secrets may raise StreamlitSecretNotFoundError if no secrets file is present
            key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
        except Exception:
            key = None

    if key:
        os.environ["OPENAI_API_KEY"] = key
        try:
            st.session_state["OPENAI_API_KEY"] = key
        except Exception:
            pass
        return key

    # Fallback: ask user to enter key in the sidebar
    with st.sidebar:
        st.warning("OpenAI API key not found. Enter it to proceed.")
        key_input = st.text_input("OPENAI_API_KEY", type="password", help="This is kept in session only unless you add it to .streamlit/secrets.toml")
        if key_input:
            os.environ["OPENAI_API_KEY"] = key_input
            try:
                st.session_state["OPENAI_API_KEY"] = key_input
            except Exception:
                pass
            st.success("API key set for this session.")
            return key_input
    return None


api_key = ensure_openai_credentials()
llm_obj = init_llm()

with st.sidebar:
    st.caption("LLM setup")
    st.write(f"Key: {_mask_key(api_key or '') if api_key else 'not set'}")
    st.write(f"LLM ready: {bool(llm_obj)}")

with st.sidebar:
    if st.button("Clear conversation & debug"):
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("_agent_bundle", None)
        try:
            from causal_agentic_ai import SESSION_STATE  # type: ignore
            SESSION_STATE["current_store_dept_list"] = None
            SESSION_STATE["last_query_scope"] = None
            SESSION_STATE["last_user_question"] = ""
        except Exception:
            pass
        st.success("Cleared past queries and context. Ask a new question.")

st.set_page_config(page_title="Causal AI Agentic Toolkit", layout="wide")
st.title("Causal AI Agentic Toolkit")
st.write("Ask a question; the agent will choose the right tool (optimal levers, what-if, or variable impact) and respond.")

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
st.header("1) Load processed dataset")

# Try to load default dataset from data/sales_data_feature_processed.csv.zip
def _load_default_from_zip() -> Optional[pd.DataFrame]:
    zip_path = ROOT.parent / "data" / "sales_data_feature_processed.csv.zip"
    if zip_path.exists():
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                csvs = [n for n in z.namelist() if n.endswith(".csv")]
                if not csvs:
                    st.warning(f"No CSV inside {zip_path}")
                    return None
                with z.open(csvs[0]) as f:
                    return pd.read_csv(f)
        except Exception as e:
            st.warning(f"Failed to read local zip: {e}")
            return None
    # fallback: try HF hub (public repo)
    try:
        hf_file = hf_hub_download(repo_id="sourish0903/causal_analysis_agenticAI", filename="data/sales_data_feature_processed.csv.zip")
        import zipfile
        with zipfile.ZipFile(hf_file, "r") as z:
            csvs = [n for n in z.namelist() if n.endswith(".csv")]
            if not csvs:
                st.warning("No CSV inside HF zip")
                return None
            with z.open(csvs[0]) as f:
                return pd.read_csv(f)
    except (RepositoryNotFoundError, RevisionNotFoundError) as e:
        st.warning(f"HuggingFace hub access error: {e}")
    except Exception as e:
        st.warning(f"HF download failed (403 or network): {e}")
    return None

# First, allow user upload as an override
uploaded_file = st.file_uploader("Upload processed CSV (expects Store, Dept, Date, Weekly_Sales, MarkDown*, etc.)", type=["csv"])

# If user didn't upload, try to load the default zip
df_loaded = None
if uploaded_file:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.success(f"Loaded dataframe with shape {df_loaded.shape} and columns: {', '.join(df_loaded.columns)}")
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Failed to read CSV: {exc}")
else:
    df_default = _load_default_from_zip()
    if df_default is not None:
        df_loaded = df_default
        st.success(f"Loaded default dataset from data/sales_data_feature_processed.csv.zip with shape {df_loaded.shape}")
    else:
        st.info("Upload a dataset to enable the tools. Tools will error if no data is set.")

if df_loaded is not None:
    try:
        set_global_data(df_loaded)
    except Exception as exc:
        st.error(f"Failed to set global data: {exc}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except Exception:
        st.warning(f"Could not parse integer from '{value}'. Ignoring.")
        return None


def parse_json_dict(value: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(value)
    except Exception as exc:
        st.error(f"Invalid JSON: {exc}")
        return None


def try_parse_json(content: str) -> Optional[Any]:
    try:
        return json.loads(content)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Agent-driven QA (no manual tool selection)
# ---------------------------------------------------------------------------
st.header("2) Ask a question")
with st.form("form_agent_query"):
    user_question = st.text_area(
        "Your question",
        value="How can I increase weekly sales by 10% in store 1 department 1?",
        height=120,
    )
    debug_mode = st.checkbox("Show debug details (tool calls, subsets, params)", value=False)
    submitted = st.form_submit_button("Ask agent")

if submitted:
    # Proceed only if a dataset is available: either uploaded by user or loaded from default zip
    if df_loaded is None:
        st.error("Please upload a dataset first or ensure the default data file exists at data/sales_data_feature_processed.csv.zip; the agent tools require data.")
    elif not user_question.strip():
        st.error("Question cannot be empty.")
    else:
        # --- INPUT GUARDRAIL: screen the question before it reaches the agent ---
        input_check = guard_input(user_question)
        if not input_check.allowed:
            st.error(f"🛡️ {input_check.message or 'Question blocked by the input guardrail.'}")
            if input_check.violations:
                st.caption("Triggered checks: " + ", ".join(input_check.violations))
            st.stop()
        safe_question = input_check.text
        if input_check.modified and input_check.violations:
            st.info("🛡️ Your question was sanitized before processing (redacted: "
                    + ", ".join(input_check.violations) + ").")

        try:
            # Conversation memory for continuity across questions
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            # Remember last question for scenario inference helpers
            set_last_user_question(safe_question)
            st.session_state.chat_messages.append(HumanMessage(content=safe_question))

            # Build or reuse agent
            if "_agent_bundle" not in st.session_state:
                st.session_state._agent_bundle = build_agent()
            agent_executor, _, _ = st.session_state._agent_bundle

            # Invoke agent with full conversation so it can rely on previous context
            state = {
                "messages": list(st.session_state.chat_messages),
                "iterations": 0,
                "final_response": "",
            }
            result = agent_executor.invoke(state)

            # Extract final reply (append to history)
            last_msg = result["messages"][-1]
            st.session_state.chat_messages = result["messages"]
            content = getattr(last_msg, "content", "(no content)")

            # --- OUTPUT GUARDRAIL: screen the answer before showing it ---
            output_check = guard_output(content, question=safe_question)
            st.subheader("Agent response")
            if not output_check.allowed:
                st.error(f"🛡️ {output_check.message or 'Response withheld by the output guardrail.'}")
                if output_check.violations:
                    st.caption("Triggered checks: " + ", ".join(output_check.violations))
            else:
                st.write(output_check.text)
                if output_check.modified and output_check.violations:
                    st.caption("🛡️ Output sanitized (redacted: "
                               + ", ".join(output_check.violations) + ").")

            if debug_mode:
                # Tool calls and arguments
                with st.expander("Debug: tool calls and outputs"):
                    tool_events = []
                    parsed_payloads = []
                    for m in result.get("messages", []):
                        tool_calls = getattr(m, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                name = getattr(tc, "name", None) or getattr(tc, "tool", None) or getattr(tc, "type", None)
                                args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
                                # If still unknown, grab dict view
                                if not name and hasattr(tc, "__dict__"):
                                    name = tc.__dict__.get("name") or tc.__dict__.get("tool") or "unknown"
                                tool_events.append({
                                    "tool": name or "unknown",
                                    "args": args if isinstance(args, dict) else str(args),
                                    "raw": getattr(tc, "__dict__", str(tc)),
                                })
                        # Collect parseable JSON payloads for dataset info
                        content_text = getattr(m, "content", None)
                        if isinstance(content_text, str):
                            parsed = try_parse_json(content_text)
                            if parsed is not None:
                                parsed_payloads.append(parsed)

                    if tool_events:
                        st.write("Tool calls detected:")
                        st.json(tool_events)
                    else:
                        st.write("No explicit tool calls detected (may still be hidden in agent response).")

                    if parsed_payloads:
                        st.write("Parsed JSON outputs (from tools):")
                        for idx, payload in enumerate(parsed_payloads, 1):
                            st.markdown(f"**Payload {idx}:**")
                            st.json(payload)
                    else:
                        st.write("No JSON payloads parsed from messages.")

                # Session info and dataset metadata
                with st.expander("Debug: session state (store/dept/question, data subsets)"):
                    try:
                        from causal_agentic_ai import SESSION_STATE, GLOBAL_DATA  # type: ignore
                        debug_info = {
                            "SESSION_STATE": SESSION_STATE,
                            "data_loaded": GLOBAL_DATA.get("df_processed") is not None,
                            "data_shape": None,
                        }
                        dfp = GLOBAL_DATA.get("df_processed")
                        if dfp is not None:
                            debug_info["data_shape"] = dfp.shape
                        st.json(debug_info)
                    except Exception as e:
                        st.write(f"Could not load debug session info: {e}")

                # Raw agent messages for full trace
                with st.expander("Debug: raw agent messages"):
                    st.json({"messages": [getattr(m, "content", str(m)) for m in result.get("messages", [])]})

        except Exception as exc:
            st.error(f"Agent error: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Causal AI Agentic Framework • Streamlit interface")
