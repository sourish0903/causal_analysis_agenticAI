"""
Golden-set evaluation harness for the causal agentic AI framework.

Runs the 15-question golden dataset (eval/golden_dataset.json) through the agent and,
for every question, computes:

  - tool-usage accuracy : did the agent route to the expected tool?
  - tool-usage success  : did the tool execute without error?
  - faithfulness        : is the answer grounded in the tool outputs?            (output vs. context)
  - relevance           : does the answer address the question given context?    (output vs. question+context)
  - answer_relevance    : does the answer respond to the query alone?            (output vs. query)

Faithfulness / relevance / answer_relevance reuse the exact LLM-as-judge scorers in
scripts/observability.py, so these numbers match what the live agent pushes to LangSmith.

LangSmith integration
---------------------
If LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is set, each agent run is traced to LangSmith
and the five scores above are attached to that run as feedback (keys: tool_usage,
tool_success, faithfulness, relevance, answer_relevance). With --upload-dataset the 15
examples are also pushed to a LangSmith dataset for record. Everything still runs and is
logged locally when LangSmith is not configured.

Outputs (under eval/results/)
  - golden_run_<timestamp>.jsonl : one structured record per question
  - golden_summary_<timestamp>.json : aggregate accuracy + mean scores (overall and per category)

Usage
  python eval/run_golden_eval.py                 # full run (needs OPENAI_API_KEY)
  python eval/run_golden_eval.py --limit 3       # first 3 questions only
  python eval/run_golden_eval.py --upload-dataset
  python eval/run_golden_eval.py --dry-run       # validate dataset + wiring, no LLM calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = Path(__file__).resolve().parent / "results"
DATASET = Path(__file__).resolve().parent / "golden_dataset.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# Dataset + data loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    valid = set(data["tools"])
    for ex in data["examples"]:
        if ex["expected_tool"] not in valid:
            raise ValueError(f"Example {ex['id']}: expected_tool {ex['expected_tool']!r} not in {valid}")
    return data


def load_processed_dataframe():
    """Load the processed retail panel (csv, then zip fallback) the tools expect."""
    import pandas as pd

    csv = ROOT / "data" / "sales_data_feature_processed.csv"
    if csv.exists():
        return pd.read_csv(csv)
    zpath = ROOT / "data" / "sales_data_feature_processed.csv.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            inner = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(inner) as fh:
                return pd.read_csv(fh)
    raise FileNotFoundError(
        "No processed dataset found at data/sales_data_feature_processed.csv[.zip]."
    )


# ---------------------------------------------------------------------------
# Trace inspection helpers
# ---------------------------------------------------------------------------
def tools_called(messages: List[Any]) -> List[str]:
    """Ordered list of tool names the agent actually invoked."""
    names: List[str] = []
    for m in messages:
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            for tc in tcs:
                n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if n:
                    names.append(n)
    return names


def any_tool_error(messages: List[Any]) -> bool:
    """True if any tool returned an error payload or the runtime flagged a tool error."""
    try:
        from langchain_core.messages import ToolMessage
    except Exception:
        ToolMessage = None
    for m in messages:
        if ToolMessage is not None and isinstance(m, ToolMessage):
            if getattr(m, "status", None) == "error":
                return True
            content = m.content if isinstance(m.content, str) else str(m.content)
            if '"status": "error"' in content or '"status":"error"' in content:
                return True
    return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the golden-set evaluation.")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N examples.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--upload-dataset", action="store_true",
                    help="Also push the golden examples to a LangSmith dataset.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate dataset + distribution without calling any LLM.")
    args = ap.parse_args()

    data = load_dataset(args.dataset)
    examples = data["examples"]
    if args.limit:
        examples = examples[: args.limit]

    # ---- dry run: just validate and report the label distribution -----------
    if args.dry_run:
        from collections import Counter
        dist = Counter(ex["category"] for ex in examples)
        print(f"Dataset OK: {len(examples)} examples")
        for cat, n in sorted(dist.items()):
            print(f"  {cat:16s} {n}")
        print("Expected tools:", sorted({ex["expected_tool"] for ex in examples}))
        print("Dry run complete (no LLM calls).")
        return

    # ---- load env + dependencies (only when actually running) ---------------
    try:
        import dotenv
        dotenv.load_dotenv(ROOT / ".env")
    except Exception:
        pass

    from langchain_core.messages import HumanMessage
    from causal_agentic_ai import build_agent, set_global_data, set_last_user_question, SESSION_STATE
    from observability import (
        init_langsmith_tracing,
        score_faithfulness,
        score_relevance,
        score_answer_relevance,
        extract_question_context_answer,
    )

    project = "causal-agentic-ai"
    tracing_on = init_langsmith_tracing(project_name=project)

    # LangSmith client + run-collection plumbing (all optional / best-effort)
    client = None
    collect_runs = None
    wait_for_all_tracers = None
    if tracing_on:
        try:
            from langsmith import Client
            from langchain_core.tracers.context import collect_runs as _collect
            from langchain_core.tracers.langchain import wait_for_all_tracers as _wait
            client, collect_runs, wait_for_all_tracers = Client(), _collect, _wait
        except Exception as e:
            print(f"⚠️ LangSmith client/tracer unavailable, continuing local-only: {e}")

    if args.upload_dataset and client is not None:
        _upload_langsmith_dataset(client, data, examples)

    # ---- build agent (observability off here: this script is the scorer) ----
    df = load_processed_dataframe()
    set_global_data(df)
    agent, llm, _ = build_agent(model=args.model, enable_observability=False)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = RESULTS / f"golden_run_{stamp}.jsonl"
    summary_path = RESULTS / f"golden_summary_{stamp}.json"

    records: List[Dict[str, Any]] = []
    print(f"\nRunning {len(examples)} golden questions (model={args.model}, "
          f"LangSmith={'on' if tracing_on else 'off'})\n" + "-" * 72)

    with open(log_path, "w") as logf:
        for ex in examples:
            # independent turn: clear any carried-over conversational scope
            SESSION_STATE["current_store_dept_list"] = None
            SESSION_STATE["last_query_scope"] = None
            SESSION_STATE["last_user_question"] = ""
            set_last_user_question(ex["question"])

            state = {"messages": [HumanMessage(content=ex["question"])], "iterations": 0, "final_response": ""}
            run_id = None
            t0 = time.time()
            if collect_runs is not None:
                with collect_runs() as cb:
                    result = agent.invoke(state)
                run_id = cb.traced_runs[0].id if getattr(cb, "traced_runs", None) else None
            else:
                result = agent.invoke(state)
            latency = time.time() - t0

            messages = result["messages"]
            question, context, answer = extract_question_context_answer(messages)
            actual = tools_called(messages)
            tool_correct = ex["expected_tool"] in actual
            tool_error = any_tool_error(messages)

            f_s, f_r = score_faithfulness(context, answer, llm)
            r_s, r_r = score_relevance(question, context, answer, llm)
            a_s, a_r = score_answer_relevance(question, answer, llm)

            # push scores to the LangSmith run as feedback
            run_url = None
            if client is not None and run_id is not None:
                try:
                    if wait_for_all_tracers:
                        wait_for_all_tracers()
                    client.create_feedback(run_id, "tool_usage", score=1.0 if tool_correct else 0.0,
                                           comment=f"expected={ex['expected_tool']} actual={actual}")
                    client.create_feedback(run_id, "tool_success", score=0.0 if tool_error else 1.0)
                    client.create_feedback(run_id, "faithfulness", score=f_s, comment=f_r)
                    client.create_feedback(run_id, "relevance", score=r_s, comment=r_r)
                    client.create_feedback(run_id, "answer_relevance", score=a_s, comment=a_r)
                    run_url = client.read_run(run_id).url
                except Exception as e:
                    print(f"  ⚠️ LangSmith feedback push failed for {ex['id']}: {e}")

            rec = {
                "id": ex["id"],
                "category": ex["category"],
                "question": ex["question"],
                "scope": ex.get("scope"),
                "expected_tool": ex["expected_tool"],
                "actual_tools": actual,
                "tool_correct": tool_correct,
                "tool_error": tool_error,
                "faithfulness": {"score": f_s, "rationale": f_r},
                "relevance": {"score": r_s, "rationale": r_r},
                "answer_relevance": {"score": a_s, "rationale": a_r},
                "answer": answer,
                "context_chars": len(context),
                "n_messages": len(messages),
                "latency_sec": round(latency, 3),
                "langsmith_run_id": str(run_id) if run_id else None,
                "langsmith_run_url": run_url,
            }
            records.append(rec)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

            mark = "✓" if tool_correct else "✗"
            print(f"  [{ex['id']}] tool {mark} ({'/'.join(actual) or 'none'})  "
                  f"faith={f_s:.2f} rel={r_s:.2f} ans_rel={a_s:.2f}  {latency:.1f}s")

    summary = summarize(records, project if tracing_on else None, str(log_path))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    print(f"\nStructured log : {log_path}")
    print(f"Summary        : {summary_path}")


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
def _mean(xs: List[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def summarize(records: List[Dict[str, Any]], project: Optional[str], log_path: str) -> Dict[str, Any]:
    def block(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(recs)
        return {
            "n": n,
            "tool_usage_accuracy": _mean([1.0 if r["tool_correct"] else 0.0 for r in recs]),
            "tool_error_rate": _mean([1.0 if r["tool_error"] else 0.0 for r in recs]),
            "mean_faithfulness": _mean([r["faithfulness"]["score"] for r in recs]),
            "mean_relevance": _mean([r["relevance"]["score"] for r in recs]),
            "mean_answer_relevance": _mean([r["answer_relevance"]["score"] for r in recs]),
        }

    by_cat: Dict[str, Any] = {}
    for cat in sorted({r["category"] for r in records}):
        by_cat[cat] = block([r for r in records if r["category"] == cat])

    # tool routing confusion: expected -> Counter(actual-or-none)
    confusion: Dict[str, Dict[str, int]] = {}
    for r in records:
        picked = r["actual_tools"][0] if r["actual_tools"] else "none"
        confusion.setdefault(r["expected_tool"], {})
        confusion[r["expected_tool"]][picked] = confusion[r["expected_tool"]].get(picked, 0) + 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": len(records),
        "overall": block(records),
        "by_category": by_cat,
        "tool_routing_confusion": confusion,
        "langsmith_project": project,
        "log_path": log_path,
    }


def _print_summary(s: Dict[str, Any]) -> None:
    o = s["overall"]
    print("\n" + "=" * 72)
    print("GOLDEN-SET SUMMARY")
    print("=" * 72)
    print(f"  questions              : {s['n']}")
    print(f"  tool-usage accuracy    : {o['tool_usage_accuracy']*100:.1f}%")
    print(f"  tool error rate        : {o['tool_error_rate']*100:.1f}%")
    print(f"  mean faithfulness      : {o['mean_faithfulness']:.3f}")
    print(f"  mean relevance         : {o['mean_relevance']:.3f}")
    print(f"  mean answer_relevance  : {o['mean_answer_relevance']:.3f}")
    print("-" * 72)
    print(f"  {'category':18s} {'n':>2s}  {'tool_acc':>8s}  {'faith':>6s}  {'rel':>5s}  {'ans_rel':>7s}")
    for cat, b in s["by_category"].items():
        print(f"  {cat:18s} {b['n']:>2d}  {b['tool_usage_accuracy']*100:>7.1f}%  "
              f"{b['mean_faithfulness']:>6.3f}  {b['mean_relevance']:>5.3f}  {b['mean_answer_relevance']:>7.3f}")
    if s.get("langsmith_project"):
        print("-" * 72)
        print(f"  LangSmith project      : {s['langsmith_project']} (scores pushed as run feedback)")


def _upload_langsmith_dataset(client, data: Dict[str, Any], examples: List[Dict[str, Any]]) -> None:
    name = data.get("name", "causal-agent-golden")
    try:
        if client.has_dataset(dataset_name=name):
            ds = client.read_dataset(dataset_name=name)
        else:
            ds = client.create_dataset(dataset_name=name, description=data.get("description", ""))
        for ex in examples:
            client.create_example(
                inputs={"question": ex["question"]},
                outputs={"expected_tool": ex["expected_tool"]},
                metadata={"id": ex["id"], "category": ex["category"], "scope": ex.get("scope")},
                dataset_id=ds.id,
            )
        print(f"Uploaded {len(examples)} examples to LangSmith dataset '{name}'.")
    except Exception as e:
        print(f"⚠️ Could not upload LangSmith dataset: {e}")


if __name__ == "__main__":
    main()
