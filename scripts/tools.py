"""
Core Tools and Helper Functions for Causal AI Agentic Framework

This module provides:
- Global state management (GLOBAL_DATA, SESSION_STATE)
- Helper utilities for data processing and LLM-assisted interpretation
- Three main @lc_tool functions for causal analysis
"""

import os
import re
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports for LangChain/LangGraph
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain.tools import tool as lc_tool
except Exception:  # pragma: no cover
    ChatOpenAI = None
    HumanMessage = None
    lc_tool = lambda f: f  # no-op decorator

# Import causal forest optimizer
from causal_forest_optimizer import (
    estimate_changes_with_causal_forest,
    compute_required_changes_from_causal_forest,
)


# ============================================================================
# GLOBAL STATE AND CONFIGURATION
# ============================================================================

def _default_model_path() -> Path:
    """Resolve comprehensive model path from env var or default location."""
    env_path = os.getenv("COMPREHENSIVE_MODEL_PATH")
    if env_path:
        return Path(env_path)
    # Fallback to repo-relative artifact path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "artifact" / "dowhy_comprehensive_model.pkl"


COMPREHENSIVE_MODEL_PATH: Path = _default_model_path()

GLOBAL_DATA: Dict[str, Any] = {
    "df_processed": None,
}

SESSION_STATE: Dict[str, Any] = {
    "current_store_dept_list": None,
    "last_query_scope": None,
    "last_user_question": None,
}

_LLM = None  # Lazily initialized ChatOpenAI instance


def init_llm(model: str = "gpt-4o-mini", temperature: float = 0) -> Optional[Any]:
    """Initialize a global LLM instance for semantic helpers."""
    global _LLM
    if ChatOpenAI is None:
        return None
    if _LLM is None:
        _LLM = ChatOpenAI(model=model, temperature=temperature)
    return _LLM


def set_global_data(df_processed: pd.DataFrame):
    """Set the global processed dataframe."""
    GLOBAL_DATA["df_processed"] = df_processed


def set_last_user_question(q: Optional[str]):
    """Store the last user question for LLM interpretation."""
    SESSION_STATE["last_user_question"] = q or ""


# ============================================================================
# MODEL AND DATA HELPERS
# ============================================================================

def load_comprehensive_model(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load the comprehensive DoWhy causal model from disk.
    
    Returns:
        dict with keys: estimates, treatments, confounders, outcome, causal_graph, dowhy_models
    """
    model_path = path or COMPREHENSIVE_MODEL_PATH
    try:
        if not Path(model_path).exists():
            print(f"⚠️ Warning: Comprehensive model not found at {model_path}")
            return None
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"⚠️ Error loading comprehensive model: {e}")
        return None


def normalize_store_dept_keys(sd_dict: Dict[str, Any]) -> Dict[str, int]:
    """Normalize flexible store/dept key naming."""
    normalized: Dict[str, int] = {}

    store_key = None
    for key in sd_dict.keys():
        key_lower = str(key).lower().strip()
        if key_lower in ["store", "store_id", "storeid", "store_num", "store_number"]:
            store_key = key
            break

    dept_key = None
    for key in sd_dict.keys():
        key_lower = str(key).lower().strip()
        if key_lower in [
            "dept", "department", "dept_id", "deptid", "department_id",
            "departmentid", "dept_num", "department_num", "department_number",
        ]:
            dept_key = key
            break

    if store_key is not None:
        normalized["store"] = int(sd_dict[store_key])
    if dept_key is not None:
        normalized["dept"] = int(sd_dict[dept_key])

    return normalized


def determine_data_subsets(store_dept_list: Optional[List[Dict]] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Determine which data subsets to analyze based on store/dept specifications.

    Args:
        store_dept_list: None | [] | [dict, ...]
            - None: Use SESSION_STATE remembered selection or entire dataset
            - []: All stores and departments (entire dataset)
            - [dict, ...]: Specific store/dept combinations with flexible key names

    Returns:
        (subsets_to_analyze, analysis_scope)
        - subsets_to_analyze: List[dict] with keys {'store','dept','data'}
        - analysis_scope: str description of scope
    """
    df = GLOBAL_DATA.get("df_processed")
    if df is None:
        raise ValueError("GLOBAL_DATA['df_processed'] is not set. Call set_global_data(df) first.")

    def make_subset(store_val, dept_val, label_store=None, label_dept=None):
        if store_val is None and dept_val is None:
            data_slice = df
        elif store_val is None:
            data_slice = df[df["Dept"] == dept_val]
        elif dept_val is None:
            data_slice = df[df["Store"] == store_val]
        else:
            data_slice = df[(df["Store"] == store_val) & (df["Dept"] == dept_val)]
        return {
            "store": label_store if label_store is not None else store_val if store_val is not None else "All",
            "dept": label_dept if label_dept is not None else dept_val if dept_val is not None else "All",
            "data": data_slice.copy(),
        }

    # CASE 1: None → Use remembered selection or entire dataset
    if store_dept_list is None:
        remembered = SESSION_STATE.get("current_store_dept_list")
        if remembered is not None:
            return determine_data_subsets(remembered)
        subsets = [make_subset(None, None)]
        scope = "Entire Dataset"
        return subsets, scope

    # CASE 2: [] → Entire dataset
    if isinstance(store_dept_list, list) and len(store_dept_list) == 0:
        subsets = [make_subset(None, None)]
        scope = "Entire Dataset"
        return subsets, scope

    # CASE 3: List of store/dept specifications
    subsets: List[Dict[str, Any]] = []
    scope_parts: List[str] = []

    for spec in store_dept_list:
        norm_spec = normalize_store_dept_keys(spec)
        store_id = norm_spec.get("store")
        dept_id = norm_spec.get("dept")

        if store_id is not None and dept_id is not None:
            subsets.append(make_subset(store_id, dept_id))
            scope_parts.append(f"Store {store_id}, Dept {dept_id}")
        elif store_id is not None:
            depts = df[df["Store"] == store_id]["Dept"].unique()
            for d in depts:
                subsets.append(make_subset(store_id, int(d)))
            scope_parts.append(f"Store {store_id} (all depts)")
        elif dept_id is not None:
            stores = df[df["Dept"] == dept_id]["Store"].unique()
            for s in stores:
                subsets.append(make_subset(int(s), dept_id))
            scope_parts.append(f"Dept {dept_id} (all stores)")
        else:
            continue

    if not subsets:
        subsets = [make_subset(None, None)]
        scope = "Entire Dataset (no valid store/dept specs)"
    else:
        scope = " + ".join(scope_parts) if scope_parts else "Specified subsets"

    return subsets, scope


# ============================================================================
# LLM-ASSISTED UTILITIES
# ============================================================================

def map_variable_name_intelligently(user_var_name: str, available_columns: List[str]) -> str:
    """Map user variable name to actual column using rules + LLM."""
    user_var_lower = str(user_var_name).lower().strip()

    # Exact match (case-insensitive)
    for col in available_columns:
        if col.lower() == user_var_lower:
            return col

    # Regex for markdown variations
    md_match = re.search(r"mark\s*down\s*(\d+)|md\s*(\d+)", user_var_lower)
    if md_match:
        num = md_match.group(1) or md_match.group(2)
        candidate = f"MarkDown{num}"
        if candidate in available_columns:
            return candidate

    # LLM semantic matching
    try:
        if _LLM is None or HumanMessage is None:
            raise RuntimeError("LLM not initialized")
        prompt = f"""Given a user's variable name and a list of available column names, identify the most likely match.

User's variable name: "{user_var_name}"

Available columns: {', '.join(available_columns)}

Common mappings to consider:
- "holiday", "is holiday" → IsHoliday
- "temp", "temperature" → Temperature
- "fuel", "fuel price", "gas price" → Fuel_Price
- "cpi", "consumer price index", "inflation" → CPI
- "unemployment", "jobless rate" → Unemployment
- "markdown", "md", "discount", "promotion" (with numbers) → MarkDown1, MarkDown2, etc.

Return ONLY the exact column name that best matches, or "NONE" if no good match exists.
Do not include explanations."""
        response = _LLM.invoke([HumanMessage(content=prompt)])
        matched_col = response.content.strip()
        if matched_col in available_columns:
            return matched_col
    except Exception as e:
        print(f"⚠️ LLM mapping failed for '{user_var_name}': {e}")

    # Fallback: string similarity
    from difflib import get_close_matches
    matches = get_close_matches(user_var_lower, [c.lower() for c in available_columns], n=1, cutoff=0.6)
    if matches:
        for col in available_columns:
            if col.lower() == matches[0]:
                return col

    return user_var_name


def parse_date_intelligently(date_str: str) -> pd.Timestamp:
    """Parse date using pandas, LLM, and dateutil fallback."""
    date_str = str(date_str).strip()

    # Remove ordinal suffixes
    date_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str, flags=re.IGNORECASE)

    # Pandas first
    try:
        return pd.to_datetime(date_str)
    except Exception:
        pass

    # LLM interpretation
    try:
        if _LLM is None or HumanMessage is None:
            raise RuntimeError("LLM not initialized")
        prompt = f"""Parse the following date string into ISO format (YYYY-MM-DD).

Date string: "{date_str}"

Return ONLY the date in YYYY-MM-DD format, nothing else.
If the date cannot be parsed, return "INVALID"."""
        response = _LLM.invoke([HumanMessage(content=prompt)])
        parsed_date_str = response.content.strip()
        if parsed_date_str != "INVALID":
            try:
                return pd.to_datetime(parsed_date_str)
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ LLM date parsing failed: {e}")

    # dateutil fallback
    try:
        from dateutil import parser as date_parser
        return pd.to_datetime(date_parser.parse(date_str, fuzzy=True))
    except Exception:
        pass

    raise ValueError(f"Could not parse date: '{date_str}' using any method")


def _infer_variable_changes_from_last_question() -> Dict[str, Dict[str, Any]]:
    """Interpret variable changes (absolute vs relative) from last user question.
    
    Returns:
        dict like {'MarkDown1': {'value': 50.0, 'type': 'absolute'}, ...}
    """
    question = SESSION_STATE.get("last_user_question") or ""
    if not question:
        return {}

    q_lower = question.lower()
    absolute_kw = ["apply", "set", "constant", "at", "fixed"]
    relative_kw = ["increase", "raise", "add", "boost", "more", "higher"]

    # Find markdown mentions with values
    matches: List[Tuple[str, str, Tuple[int, int]]] = []
    pattern_var_then_value = re.compile(r"mark\s*down\s*(\d)\D{0,10}?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    pattern_value_then_var = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%?\s*(?:mark\s*down\s*(\d))", re.IGNORECASE)

    for m in pattern_var_then_value.finditer(question):
        matches.append((m.group(1), m.group(2), m.span()))
    for m in pattern_value_then_var.finditer(question):
        matches.append((m.group(2), m.group(1), m.span()))

    if not matches:
        return {}

    # Try LLM classification first
    change_types: Dict[str, str] = {}
    if _LLM is not None and HumanMessage is not None:
        interpretation_prompt = f"""Analyze the user's intent for markdown changes:

"{question}"

For EACH markdown variable mentioned, determine: ABSOLUTE (set to X%) or RELATIVE (increase by X%)?

Return ONLY: {{"MarkDown1": "absolute", "MarkDown2": "relative"}}"""
        try:
            response = _LLM.invoke([HumanMessage(content=interpretation_prompt)])
            change_types = json.loads(response.content.strip())
        except Exception:
            change_types = {}

    # Local keyword fallback
    result: Dict[str, Dict[str, Any]] = {}
    for num, val, span in matches:
        var_name = f"MarkDown{num}"
        change_type = (change_types.get(var_name) or "").lower()

        # Inspect local context around the match
        window = 20
        left = max(0, span[0] - window)
        right = min(len(q_lower), span[1] + window)
        ctx = q_lower[left:right]
        if any(k in ctx for k in absolute_kw):
            change_type = "absolute"
        elif any(k in ctx for k in relative_kw):
            change_type = "relative"
        if not change_type:
            change_type = "relative"

        try:
            value = float(val)
        except Exception:
            continue

        result[var_name] = {"value": value, "type": change_type}

    return result


# ============================================================================
# CORE TOOLS (EXPORTED)
# -----------------------------
@lc_tool
def find_optimal_levers(
    target_percentage: float,
    store: Optional[int] = None,
    dept: Optional[int] = None,
    analyze_all_data: bool = False,
    max_levers: int = 2,
    top_n: int = 3
) -> str:
    """Find optimal lever/influencing factors (markdowns) combinations to achieve a target sales increase percentage. Here lever means any treatments like MarkDown1-5,which affects sales.
    So this tool may be used even if there is no explicit mention of 'lever' in the query. Use judgement whether the user is asking for influencing a sales target.
    Concatenates multiple sub-datasets into a single dataset for unified analysis.
    
    ⚠️ CRITICAL - PARAMETER EXTRACTION:
    When user says "store 1 department 1" → call with store=1, dept=1
    When user says "store 5" → call with store=5, dept=None
    When user says "all data" → call with analyze_all_data=True
    
    EXAMPLES:
    Query: "increase sales by 25% at store 1 department 1"
    Call: find_optimal_levers(target_percentage=25, store=1, dept=1)
    
    Query: "20% increase for store 5"
    Call: find_optimal_levers(target_percentage=20, store=5, dept=None)
    
    Query: "30% increase across all data"
    Call: find_optimal_levers(target_percentage=30, analyze_all_data=True)
    
    Args:
        target_percentage: Target sales increase percentage (e.g., 20 for 20% increase)
        store: Store number extracted from user query (e.g., 1, 5, 10). REQUIRED if user mentions it.
        dept: Department number extracted from user query (e.g., 1, 5, 10). REQUIRED if user mentions it.
        analyze_all_data: Set to True if user asks for "all data", "entire dataset", "all stores and departments"
        max_levers: Maximum number of levers in a combination (1-3)
        top_n: Number of top solutions to return
        
    Returns:
        JSON string with recommended lever combinations and their details
    """
    try:
        # Convert simple parameters to store_dept_list format
        # Support dept across all stores when analyze_all_data is True
        if analyze_all_data:
            if dept is not None and store is None:
                store_dept_list = [{"dept": dept}]
            elif store is not None and dept is None:
                store_dept_list = [{"store": store}]
            elif store is not None and dept is not None:
                store_dept_list = [{"store": store, "dept": dept}]
            else:
                store_dept_list = []
        elif store is not None or dept is not None:
            # Specific store/dept combination
            store_dept_list = [{}]
            if store is not None:
                store_dept_list[0]['store'] = store
            if dept is not None:
                store_dept_list[0]['dept'] = dept
        else:
            # Use session memory
            store_dept_list = None
        
        # 💾 SAVE USER INPUT: Store the original user's selection BEFORE calling determine_data_subsets
        # This is critical - we must save what the USER explicitly requested, not what the session had
        original_user_input = store_dept_list
        
        # Determine which data subsets to analyze
        subsets_to_analyze, analysis_scope = determine_data_subsets(store_dept_list)
        
        # 💾 PERSISTENCE: Save current selection to SESSION_STATE for follow-up queries
        # CRITICAL: Only save if user provided explicit parameters (not using previous context)
        # If original_user_input is None, it means we're using session memory - don't overwrite it
        if original_user_input is not None:
            # User provided explicit parameters - save them
            SESSION_STATE['current_store_dept_list'] = original_user_input
            SESSION_STATE['last_query_scope'] = analysis_scope
        
        # 🐛 DEBUG: Show which dataset is being used
        print("\n" + "="*70)
        print("📊 DATASET SCOPE DEBUG (find_optimal_levers)")
        print("="*70)
        if store_dept_list is None:
            print("📋 Using PREVIOUS CONTEXT from session memory")
            print(f"   Stored selection: {SESSION_STATE['current_store_dept_list']}")
        else:
            print(f"📋 Using parameters: {store_dept_list}")
        print(f"\n🎯 Analysis Scope: {analysis_scope}")
        print(f"📦 Number of subsets: {len(subsets_to_analyze)}")
        total_rows = sum(len(s['data']) for s in subsets_to_analyze)
        print(f"📊 Total data rows: {total_rows}")
        print("="*70 + "\n")
        
        # Concatenate all subsets into a single dataset for analysis
        combined_data = pd.concat([s['data'] for s in subsets_to_analyze], ignore_index=True)
        
        if combined_data.empty:
            return json.dumps({"status": "error", "message": "No data found for specified selection"})
        
        # Use the combined dataset with causal forest
        results_cf, levers = estimate_changes_with_causal_forest(
            store=combined_data['Store'].iloc[0] if 'Store' in combined_data.columns else 1,
            dept=combined_data['Dept'].iloc[0] if 'Dept' in combined_data.columns else 1,
            levers_all=True,
            data=combined_data,
            max_lever_combo=max_levers
        )
        
        candidates = compute_required_changes_from_causal_forest(
            results=results_cf,
            target_pct=target_percentage,
            levers=levers,
            top_n=top_n,
            ensure_multi_lever=(max_levers > 1)
        )
        
        if candidates.empty:
            return json.dumps({"status": "no_solutions", "message": "No feasible solutions found"})
        
        # Convert to dict for JSON serialization
        results = []
        for _, row in candidates.head(top_n).iterrows():
            results.append({
                "levers": row['combo'],
                "num_levers": int(row['num_levers']),
                "sample_size": int(row['n']),
                "description": row['description'],
                "changes": str(row['required_change'])
            })
        
        return json.dumps({
            "status": "success",
            "analysis_scope": analysis_scope,
            "data_points_analyzed": len(combined_data),
            "target_pct": target_percentage,
            "solutions": results,
            "message": f"Found {len(results)} optimal solutions for {analysis_scope}"
        }, indent=2)
    
    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})


@lc_tool
def analyze_variable_impact(
    variables: List[str],
    store: Optional[int] = None,
    dept: Optional[int] = None,
    analyze_all_data: bool = False,
) -> str:
    """Analyze causal impact of one or more variables on weekly sales using DoWhy."""
    import json

    try:
        if analyze_all_data:
            store_dept_list = []
        elif store is not None or dept is not None:
            store_dept_list = [{}]
            if store is not None:
                store_dept_list[0]["store"] = store
            if dept is not None:
                store_dept_list[0]["dept"] = dept
        else:
            store_dept_list = None

        original_user_input = store_dept_list
        subsets_to_analyze, analysis_scope = determine_data_subsets(store_dept_list)

        if original_user_input is not None:
            SESSION_STATE["current_store_dept_list"] = original_user_input
            SESSION_STATE["last_query_scope"] = analysis_scope

        all_data = pd.concat([s["data"] for s in subsets_to_analyze], ignore_index=True)
        if all_data.empty:
            return json.dumps({"status": "error", "message": "No data found for specified selection"})

        available_columns = list(all_data.columns)

        mapped_variables: List[str] = []
        mapping_log: List[Dict[str, Any]] = []
        for var_name in variables:
            mapped_name = map_variable_name_intelligently(var_name, available_columns)
            mapped_variables.append(mapped_name)
            mapping_log.append({"user_input": var_name, "mapped_to": mapped_name, "found": mapped_name in available_columns})

        results = []
        for var_name, mapping in zip(mapped_variables, mapping_log):
            if var_name not in all_data.columns:
                results.append(
                    {
                        "variable": mapping["user_input"],
                        "mapped_to": var_name,
                        "status": "error",
                        "message": f"Variable '{var_name}' not found in dataset",
                    }
                )
                continue

            try:
                median_val = all_data[var_name].median()
                all_data["treatment"] = (all_data[var_name] > median_val).astype(int)

                # Confounders: all except treatment/outcome/ids/date
                confounders = [
                    col
                    for col in all_data.columns
                    if col not in [var_name, "Weekly_Sales", "treatment", "Date", "Store", "Dept"]
                ]

                # Build and run DoWhy model
                from dowhy import CausalModel  # imported here to avoid hard dependency at import-time

                causal_graph = f"""
                digraph {{
                    treatment -> Weekly_Sales;
                    {'; '.join([f'{conf} -> treatment; {conf} -> Weekly_Sales' for conf in confounders[:5]])};
                }}
                """

                model = CausalModel(data=all_data, treatment="treatment", outcome="Weekly_Sales", graph=causal_graph)
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

                mean_sales = all_data["Weekly_Sales"].mean()
                impact_percentage = (estimate.value / mean_sales) * 100 if mean_sales != 0 else 0

                results.append(
                    {
                        "variable": mapping["user_input"],
                        "mapped_to": var_name,
                        "status": "success",
                        "causal_effect": float(estimate.value),
                        "mean_weekly_sales": float(mean_sales),
                        "impact_percentage": float(impact_percentage),
                        "median_threshold": float(median_val),
                        "interpretation": f"Increasing {var_name} above median ({median_val:.2f}) causes {'an increase' if estimate.value > 0 else 'a decrease'} of ${abs(estimate.value):.2f} in weekly sales ({abs(impact_percentage):.2f}% {'increase' if estimate.value > 0 else 'decrease'})",
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "variable": mapping["user_input"],
                        "mapped_to": var_name,
                        "status": "error",
                        "message": f"Causal analysis failed: {str(e)}",
                    }
                )

        summary = {
            "status": "success",
            "analysis_scope": analysis_scope,
            "total_records": len(all_data),
            "variable_mapping": mapping_log,
            "causal_impacts": results,
        }
        return json.dumps(summary, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
    

@lc_tool
def analyze_whatif_scenario(
    variable_changes: Optional[Dict[str, float]] = None,
    store: Optional[int] = None,
    dept: Optional[int] = None,
    analyze_all_data: bool = False,
    date: Optional[str] = None,
) -> str:
    """Run a what-if analysis with optional store/dept/date scoping.

    - Uses LLM to interpret user intent (absolute vs relative value changes)
    - Uses the saved DoWhy comprehensive model to estimate sales impact when available.
    - If a date is provided, restrict each store/dept subset to that date and use only the first matching row.
    - Persists explicit store/dept choices into SESSION_STATE so follow-ups reuse the same scope.
    - Returns JSON with per-segment impacts and requested change percentages.
    - When absolute MarkDown changes applied, automatically sets corresponding _applied column to 1 (impact excluded).
    """
    import json

    try:
        # Keep original user-provided changes; infer types from last question separately
        interpreted_types = _infer_variable_changes_from_last_question()

        if not variable_changes and not interpreted_types:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No variable_changes provided or inferred. Please specify something like {'MarkDown1': 10}.",
                    "analysis_scope": "unspecified",
                }
            )

        comprehensive_model = load_comprehensive_model()
        model_estimates = comprehensive_model.get("estimates", {}) if comprehensive_model else {}
        model_treatments = set(comprehensive_model.get("treatments", [])) if comprehensive_model else set()

        # Scope conversion (support dept across all stores, or store across all depts)
        if analyze_all_data:
            if dept is not None and store is None:
                store_dept_list = [{"dept": dept}]
            elif store is not None and dept is None:
                store_dept_list = [{"store": store}]
            elif store is not None and dept is not None:
                store_dept_list = [{"store": store, "dept": dept}]
            else:
                store_dept_list = []  # full dataset
        elif store is not None or dept is not None:
            store_dept_list = [{}]
            if store is not None:
                store_dept_list[0]["store"] = store
            if dept is not None:
                store_dept_list[0]["dept"] = dept
        else:
            store_dept_list = None

        original_user_input = store_dept_list
        subsets_to_analyze, analysis_scope = determine_data_subsets(store_dept_list)
        if original_user_input is not None:
            SESSION_STATE["current_store_dept_list"] = original_user_input
            SESSION_STATE["last_query_scope"] = analysis_scope

        parsed_date = None
        if date:
            parsed_date = parse_date_intelligently(date)

        filtered_subsets = []
        for subset in subsets_to_analyze:
            data = subset["data"].copy()
            if parsed_date is not None:
                if not np.issubdtype(data["Date"].dtype, np.datetime64):
                    data["Date"] = pd.to_datetime(data["Date"])
                data = data[data["Date"].dt.date == parsed_date.date()]
                if data.empty:
                    continue
                data = data.head(1)
            filtered_subsets.append({**subset, "data": data})

        if not filtered_subsets:
            return json.dumps(
                {"status": "error", "message": "No data found for the requested scope/date", "analysis_scope": analysis_scope}
            )

        scenario_type = "date-specific" if parsed_date is not None else "aggregate"

        results_by_segment = []
        for subset in filtered_subsets:
            data = subset["data"]
            baseline_sales = float(data["Weekly_Sales"].mean()) if not data.empty else 0.0
            segment_predicted = baseline_sales
            variable_impacts = []

            # Build a normalized change spec per variable using mapped names and interpreted types
            available_cols = list(data.columns)
            iter_items = variable_changes.items() if isinstance(variable_changes, dict) else []
            for user_var, change_spec in iter_items:
                mapped_var = map_variable_name_intelligently(user_var, available_cols)

                # Extract value and type, prefer interpreted type from question if available
                if isinstance(change_spec, dict):
                    change_value = float(change_spec.get("value", 0.0))
                    change_type = str(change_spec.get("type", "")).lower()
                else:
                    change_value = float(change_spec)
                    change_type = ""

                # If no explicit type, try from interpreted types using mapped name
                if not change_type:
                    key_for_types = mapped_var
                    if key_for_types in interpreted_types:
                        change_type = str(interpreted_types[key_for_types].get("type", "relative")).lower()
                    else:
                        change_type = "relative"

                valid_values = data[mapped_var].replace([np.inf, -np.inf], np.nan).dropna()
                if valid_values.empty:
                    current_value = 0.0
                elif scenario_type == "date-specific":
                    current_value = float(valid_values.iloc[0])
                else:
                    current_value = float(valid_values.mean())

                # Calculate new value
                if change_type == "absolute":
                    # Treat negatives as absolute magnitude
                    change_value = abs(change_value)
                    if change_value > 1:
                        new_value = change_value / 100.0
                        requested_pct_display = f"→ {change_value}%"
                    else:
                        new_value = change_value
                        requested_pct_display = f"→ {change_value*100:.1f}%"

                    # Update _applied column (no impact contribution)
                    if "MarkDown" in mapped_var and not mapped_var.endswith("_applied"):
                        applied_col = f"{mapped_var}_applied"
                        if applied_col in data.columns:
                            new_applied_value = 1.0 if new_value > 0 else 0.0
                            data[applied_col] = new_applied_value
                else:
                    requested_pct = float(change_value / 100 if abs(change_value) > 1 else change_value)
                    new_value = current_value * (1 + requested_pct)
                    # Render with sign correctly
                    sign = "+" if requested_pct >= 0 else ""
                    requested_pct_display = f"{sign}{requested_pct*100:.1f}%"

                delta = new_value - current_value

                # Clamp binary
                if mapped_var.endswith("_applied"):
                    new_value = min(max(new_value, 0.0), 1.0)

                impact_method = "correlation_fallback"
                impact = delta

                if mapped_var in model_estimates:
                    beta = float(model_estimates[mapped_var])
                    impact = beta * delta
                    impact_method = "dowhy_comprehensive"
                elif mapped_var in model_treatments:
                    impact_method = "dowhy_comprehensive_missing_estimate"

                segment_predicted += impact

                variable_impacts.append(
                    {
                        "variable": user_var,
                        "mapped_to": mapped_var,
                        "current_value": current_value,
                        "new_value": new_value,
                        "delta": delta,
                        "requested_change_pct": requested_pct_display,
                        "change_type": change_type,
                        "impact_on_sales": impact,
                        "impact_method": impact_method,
                        "causal_beta": model_estimates.get(mapped_var, "N/A"),
                    }
                )

            segment_metadata = {
                "store": subset.get("store", "All"),
                "dept": subset.get("dept", "All"),
                "num_rows": len(data),
                "baseline_sales": baseline_sales,
                "predicted_sales": segment_predicted,
                "total_impact": segment_predicted - baseline_sales,
                "variable_impacts": variable_impacts,
            }
            results_by_segment.append(segment_metadata)

        return json.dumps(
            {
                "status": "success",
                "analysis_scope": analysis_scope,
                "scenario_type": scenario_type,
                "model_available": comprehensive_model is not None,
                "results_by_segment": results_by_segment,
            },
            indent=2,
        )

    except Exception as e:
        import traceback
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})


__all__ = [
    # State management
    "GLOBAL_DATA",
    "SESSION_STATE",
    "init_llm",
    "set_global_data",
    "set_last_user_question",
    # Config
    "COMPREHENSIVE_MODEL_PATH",
    # Model and data helpers
    "load_comprehensive_model",
    "normalize_store_dept_keys",
    "determine_data_subsets",
    # LLM utilities
    "map_variable_name_intelligently",
    "parse_date_intelligently",
    # Tools
    "analyze_whatif_scenario",
    "find_optimal_levers",
    "analyze_variable_impact",
]
