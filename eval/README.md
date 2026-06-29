---
noteId: "db337fc0706111f1a6b58d58207ce531"
tags: []

---

# Golden-set evaluation

A 15-question golden dataset and a runner that scores the causal agent on tool routing
and answer quality, wired into the LangSmith observability framework.

## Files

```
eval/
├── golden_dataset.json     # 15 labelled questions (5 optimal-lever, 5 what-if, 5 variable-impact)
├── run_golden_eval.py      # runner: executes the agent, scores it, logs + pushes to LangSmith
└── results/                # generated: golden_run_<ts>.jsonl + golden_summary_<ts>.json
```

## What it measures

For each question the runner records:

| metric | compares | source |
|---|---|---|
| **tool_usage accuracy** | expected vs. actual tool the agent called | golden label vs. trace |
| **tool_success** | did the tool run without an error payload | trace |
| **faithfulness** | answer vs. tool-output context | `score_faithfulness` |
| **relevance** | answer vs. question *given context* | `score_relevance` |
| **answer_relevance** | answer vs. the query *alone* | `score_answer_relevance` |

The three judge scores reuse the exact scorers in `scripts/observability.py`, so the
numbers match what the live agent streams to LangSmith.

## Running

Use the interpreter that has the agent stack (langgraph / langchain-openai / langsmith) —
in this repo that is the `causal_agent` venv. Requires `OPENAI_API_KEY` (in `.env`).

```bash
# validate the dataset and wiring without spending any tokens
causal_agent/bin/python eval/run_golden_eval.py --dry-run

# full run (15 questions; ~1 agent call + 3 judge calls each)
causal_agent/bin/python eval/run_golden_eval.py

# a quick smoke test on the first 3 questions
causal_agent/bin/python eval/run_golden_eval.py --limit 3

# also register the examples as a LangSmith dataset
causal_agent/bin/python eval/run_golden_eval.py --upload-dataset
```

## LangSmith integration

If `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`) is set, every agent run is traced and the
five scores above are attached to that run as **feedback** (keys: `tool_usage`,
`tool_success`, `faithfulness`, `relevance`, `answer_relevance`). The runner captures the
root run id via `collect_runs()` and pushes feedback with the LangSmith `Client`, so the
golden scores show up alongside the live-traffic ones in the
`causal-agentic-ai` project. With `--upload-dataset` the 15 examples are also written to a
LangSmith dataset named `causal-agent-golden`.

If LangSmith is not configured the runner still executes and writes the local logs; it just
skips the feedback push.

## Outputs

- **`golden_run_<timestamp>.jsonl`** — one structured record per question: the question,
  expected/actual tools, the three judge scores with their one-line rationales, the final
  answer, latency, and the LangSmith run id/url.
- **`golden_summary_<timestamp>.json`** — overall and per-category tool-usage accuracy,
  tool error rate, and mean faithfulness / relevance / answer_relevance, plus a
  tool-routing confusion map (expected → which tool was actually picked).

A summary table is also printed to the console at the end of the run.

## Note on the scores

Tool-usage accuracy and tool-success are objective (label vs. trace). The faithfulness /
relevance / answer_relevance scores are LLM-as-judge **triage signals**, not a verdict on
the *correctness* of the underlying causal estimate — validating that requires a
human-in-the-loop domain expert (see the paper's §4.3 and Limitations).
