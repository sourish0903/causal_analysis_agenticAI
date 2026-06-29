"""
LangSmith-based observability for the causal agentic AI framework.

Provides real-time auditing for an agent run:
- Tool usage: every tool call (name, latency, success/error) is logged as LangSmith
  run feedback the moment it happens.
- Latency & cost: per-LLM-call wall-clock latency and token usage are captured,
  converted to an approximate USD cost via a configurable per-model price table,
  accumulated in-process (AgenticObservabilityHandler.metrics) and pushed as
  LangSmith feedback both per LLM call and per top-level run.
- Faithfulness: LLM-as-judge score for whether the final answer's claims are
  supported by the tool-output context the agent actually used.
- Relevance: LLM-as-judge score for whether the final answer addresses the
  user's question given that same context.

Wired in via AgenticObservabilityHandler, a LangChain callback handler bound as a
default callback on the compiled agent graph (see causal_agentic_ai.build_agent).
Scoring and feedback only run when LangSmith tracing is actually configured
(LANGSMITH_API_KEY / LANGCHAIN_API_KEY env var present) so there is no cost or
behavior change when observability isn't set up.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover
    AIMessage = ToolMessage = HumanMessage = None
    BaseCallbackHandler = object

try:
    from langsmith import Client as LangSmithClient
except Exception:  # pragma: no cover
    LangSmithClient = None

from tools import init_llm


# ============================================================================
# TRACING SETUP
# ============================================================================

_TRACING_ACTIVE = False
_LANGSMITH_CLIENT: Optional[Any] = None


def init_langsmith_tracing(project_name: str = "causal-agentic-ai") -> bool:
    """Enable LangSmith tracing for this process if an API key is configured.

    Returns True if tracing is active, False otherwise (the agent still runs
    normally; it just won't be audited).
    """
    global _TRACING_ACTIVE
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key or LangSmithClient is None:
        _TRACING_ACTIVE = False
        return False

    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", project_name)
    _TRACING_ACTIVE = True
    return True


def _get_langsmith_client() -> Optional[Any]:
    global _LANGSMITH_CLIENT
    if not _TRACING_ACTIVE or LangSmithClient is None:
        return None
    if _LANGSMITH_CLIENT is None:
        try:
            _LANGSMITH_CLIENT = LangSmithClient()
        except Exception as e:
            print(f"⚠️ LangSmith client init failed: {e}")
            return None
    return _LANGSMITH_CLIENT


# ============================================================================
# MESSAGE EXTRACTION
# ============================================================================

def extract_question_context_answer(messages: List[Any]) -> Tuple[str, str, str]:
    """From a LangGraph agent's message history, pull the latest user question,
    the concatenated tool outputs (context), and the final AI answer."""
    question = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            question = m.content
            break

    context = "\n\n".join(
        f"[{getattr(m, 'name', None) or 'tool'}] {m.content}"
        for m in messages
        if isinstance(m, ToolMessage)
    )

    answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            answer = m.content
            break

    return question, context, answer


# ============================================================================
# LLM-AS-JUDGE SCORERS
# ============================================================================

_FAITHFULNESS_PROMPT = """You are auditing an AI agent's answer for factual faithfulness to its supporting context.

Context (tool outputs the agent had access to):
{context}

Agent's final answer:
{answer}

Score how faithful the answer is to the context on a 0.0-1.0 scale:
- 1.0 = every factual claim in the answer is directly supported by the context
- 0.0 = the answer contradicts or invents facts not present in the context
Penalize hallucinated numbers, store/dept IDs, or causal claims not present in the context.

Return ONLY valid JSON: {{"score": <float 0-1>, "rationale": "<one short sentence>"}}"""

_RELEVANCE_PROMPT = """You are auditing whether an AI agent's answer actually addresses the user's question, given the context it used.

User's question:
{question}

Context (tool outputs the agent had access to):
{context}

Agent's final answer:
{answer}

Score relevance on a 0.0-1.0 scale:
- 1.0 = the answer directly and completely addresses what the user asked
- 0.0 = the answer is off-topic or ignores the question

Return ONLY valid JSON: {{"score": <float 0-1>, "rationale": "<one short sentence>"}}"""

_ANSWER_RELEVANCE_PROMPT = """You are auditing whether an AI agent's answer responds to the user's question, judged against the QUERY ALONE (ignore any external context).

User's question:
{question}

Agent's final answer:
{answer}

Score answer relevance on a 0.0-1.0 scale, considering only the query and the answer:
- 1.0 = the answer directly responds to exactly what was asked, covering every part of a multi-part request
- 0.0 = the answer is off-topic, evasive, or drops what was asked
Penalize answers that are internally coherent but do not actually respond to the query, or that silently omit part of the request.

Return ONLY valid JSON: {{"score": <float 0-1>, "rationale": "<one short sentence>"}}"""


def _llm_judge(prompt: str, llm: Optional[Any]) -> Tuple[float, str]:
    if llm is None or HumanMessage is None:
        return 0.0, "No judge LLM available; score not computed."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        rationale = str(parsed.get("rationale", ""))[:500]
        return score, rationale
    except Exception as e:
        return 0.0, f"Judge scoring failed: {e}"


def score_faithfulness(context: str, answer: str, llm: Optional[Any] = None) -> Tuple[float, str]:
    """LLM-as-judge: are the answer's claims supported by the tool-output context?"""
    if not context.strip():
        return 1.0, "No tool context was used; treated as trivially faithful."
    judge_llm = llm or init_llm()
    prompt = _FAITHFULNESS_PROMPT.format(context=context[:6000], answer=answer[:3000])
    return _llm_judge(prompt, judge_llm)


def score_relevance(question: str, context: str, answer: str, llm: Optional[Any] = None) -> Tuple[float, str]:
    """LLM-as-judge: does the answer address the user's question given the context used?"""
    judge_llm = llm or init_llm()
    prompt = _RELEVANCE_PROMPT.format(question=question[:2000], context=context[:6000], answer=answer[:3000])
    return _llm_judge(prompt, judge_llm)


def score_answer_relevance(question: str, answer: str, llm: Optional[Any] = None) -> Tuple[float, str]:
    """LLM-as-judge: does the answer respond to the query, judged against the question alone
    (no context)? Catches answers that are grounded yet off-topic, or that drop part of a
    multi-part request.

    NOTE: This is a triage/screening signal, not a verdict on factual correctness. Assessing
    whether a causal estimate is actually right requires a human-in-the-loop domain expert.
    """
    judge_llm = llm or init_llm()
    prompt = _ANSWER_RELEVANCE_PROMPT.format(question=question[:2000], answer=answer[:3000])
    return _llm_judge(prompt, judge_llm)


# ============================================================================
# COST & TOKEN ACCOUNTING
# ============================================================================

# Approximate OpenAI list prices in USD per 1K tokens as (input, output).
# Extend or override at runtime by mutating MODEL_PRICING, or set a global default
# via the OBSERVABILITY_PRICE_INPUT / OBSERVABILITY_PRICE_OUTPUT env vars.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "gpt-4.1": (0.00200, 0.00800),
    "gpt-4.1-mini": (0.00040, 0.00160),
    "gpt-4.1-nano": (0.00010, 0.00040),
    "gpt-4-turbo": (0.01000, 0.03000),
    "gpt-3.5-turbo": (0.00050, 0.00150),
}


def _default_pricing() -> Tuple[float, float]:
    try:
        return (
            float(os.getenv("OBSERVABILITY_PRICE_INPUT", "0.00015")),
            float(os.getenv("OBSERVABILITY_PRICE_OUTPUT", "0.00060")),
        )
    except (TypeError, ValueError):
        return (0.00015, 0.00060)


def lookup_pricing(model: Optional[str]) -> Tuple[float, float]:
    """Return (input, output) price per 1K tokens for a model name.

    Falls back to the longest known prefix match (so a dated alias like
    "gpt-4o-mini-2024-07-18" still resolves to "gpt-4o-mini"), then to the
    env-configurable default.
    """
    if model:
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]
        matches = [k for k in MODEL_PRICING if model.startswith(k)]
        if matches:
            return MODEL_PRICING[max(matches, key=len)]
    return _default_pricing()


def estimate_cost_usd(model: Optional[str], prompt_tokens: int, completion_tokens: int) -> float:
    """Approximate USD cost of an LLM call given its token counts."""
    in_rate, out_rate = lookup_pricing(model)
    return (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate


def _extract_token_usage(response: Any) -> Tuple[int, int, int]:
    """Pull (prompt, completion, total) token counts from an LLMResult, tolerant
    of LangChain version differences (llm_output.token_usage vs. per-generation
    usage_metadata)."""
    prompt = completion = total = 0

    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or (prompt + completion))

    if total == 0:  # fallback to newer langchain_core usage_metadata on messages
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    meta = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                    if meta:
                        prompt += int(meta.get("input_tokens", 0) or 0)
                        completion += int(meta.get("output_tokens", 0) or 0)
                        total += int(meta.get("total_tokens", 0) or 0)
        except Exception:
            pass
        if total == 0:
            total = prompt + completion

    return prompt, completion, total


def _extract_model_name(response: Any) -> Optional[str]:
    llm_output = getattr(response, "llm_output", None) or {}
    return llm_output.get("model_name") or llm_output.get("model")


# ============================================================================
# CALLBACK HANDLER (real-time tool-usage auditing + latency/cost + per-turn scoring)
# ============================================================================

class AgenticObservabilityHandler(BaseCallbackHandler):
    """LangChain callback handler that audits tool usage in real time, monitors
    per-LLM-call latency and token cost, and scores faithfulness/relevance on the
    agent's final answer, pushing all of these as LangSmith run feedback.

    Latency/token/cost figures are always accumulated in-process (``self.metrics``,
    readable via ``get_metrics()``) so they are useful even without LangSmith;
    feedback pushes and LLM-as-judge scoring no-op when tracing isn't configured.
    """

    def __init__(self, judge_llm: Optional[Any] = None, model_name: str = "gpt-4o-mini"):
        super().__init__()
        self._tool_starts: Dict[str, Dict[str, Any]] = {}
        self._llm_starts: Dict[str, float] = {}
        self._run_starts: Dict[str, Dict[str, Any]] = {}
        self._judge_llm = judge_llm
        self._model_name = model_name  # used when a response omits its model name
        self.metrics: Dict[str, Any] = self._new_metrics()

    # -- in-process metric accumulator --------------------------------------
    @staticmethod
    def _new_metrics() -> Dict[str, Any]:
        return {
            "runs": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "llm_latency_sec": 0.0,
        }

    def reset_metrics(self) -> None:
        """Zero the cumulative latency/token/cost counters."""
        self.metrics = self._new_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """Return a rounded snapshot of cumulative metrics with derived averages."""
        m = dict(self.metrics)
        m["cost_usd"] = round(m["cost_usd"], 6)
        m["llm_latency_sec"] = round(m["llm_latency_sec"], 3)
        m["avg_llm_latency_sec"] = round(m["llm_latency_sec"] / m["llm_calls"], 3) if m["llm_calls"] else 0.0
        return m

    # -- real-time tool usage auditing --------------------------------------
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs) -> None:
        self.metrics["tool_calls"] += 1
        self._tool_starts[str(run_id)] = {
            "name": (serialized or {}).get("name", "unknown_tool"),
            "start": time.time(),
            "input": input_str,
        }

    def _push_tool_feedback(self, run_id: Any, status: str, output_preview: str = "", error: str = "") -> None:
        info = self._tool_starts.pop(str(run_id), {})
        client = _get_langsmith_client()
        if client is None:
            return
        latency = time.time() - info.get("start", time.time())
        comment = json.dumps({
            "tool": info.get("name", "unknown_tool"),
            "latency_sec": round(latency, 3),
            "input": (info.get("input") or "")[:500],
            "output_preview": output_preview[:500],
            "error": error,
        })
        try:
            client.create_feedback(run_id=run_id, key="tool_usage", score=1.0 if status == "success" else 0.0, comment=comment)
        except Exception as e:
            print(f"⚠️ Failed to push tool_usage feedback to LangSmith: {e}")

    def on_tool_end(self, output: Any, *, run_id, parent_run_id=None, tags=None, **kwargs) -> None:
        self._push_tool_feedback(run_id, status="success", output_preview=str(output))

    def on_tool_error(self, error: BaseException, *, run_id, parent_run_id=None, tags=None, **kwargs) -> None:
        self._push_tool_feedback(run_id, status="error", error=str(error))

    # -- latency & token-cost monitoring (per LLM call) ---------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs) -> None:
        self._llm_starts[str(run_id)] = time.time()

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs) -> None:
        # Fallback for stacks that emit on_llm_start instead of on_chat_model_start.
        self._llm_starts.setdefault(str(run_id), time.time())

    def on_llm_end(self, response: Any, *, run_id, parent_run_id=None, **kwargs) -> None:
        start = self._llm_starts.pop(str(run_id), None)
        latency = time.time() - start if start is not None else 0.0
        prompt_tokens, completion_tokens, total_tokens = _extract_token_usage(response)
        model = _extract_model_name(response) or self._model_name
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

        m = self.metrics
        m["llm_calls"] += 1
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["total_tokens"] += total_tokens
        m["cost_usd"] += cost
        m["llm_latency_sec"] += latency

        client = _get_langsmith_client()
        if client is None:
            return
        comment = json.dumps({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
            "latency_sec": round(latency, 3),
        })
        try:
            client.create_feedback(run_id=run_id, key="llm_latency_sec", score=round(latency, 3), comment=comment)
            client.create_feedback(run_id=run_id, key="llm_cost_usd", score=round(cost, 6), comment=comment)
            client.create_feedback(run_id=run_id, key="token_usage", score=float(total_tokens), comment=comment)
        except Exception as e:
            print(f"⚠️ Failed to push latency/cost feedback to LangSmith: {e}")

    # -- per-run aggregation start marker -----------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs) -> None:
        if parent_run_id is None:  # only snapshot the top-level agent run
            self._run_starts[str(run_id)] = {
                "t": time.time(),
                "tokens": self.metrics["total_tokens"],
                "cost": self.metrics["cost_usd"],
                "llm_calls": self.metrics["llm_calls"],
            }

    # -- faithfulness / relevance scoring on the final answer ---------------
    # These are automated triage signals, not a verdict on correctness: certifying
    # that a causal estimate is actually right needs a human-in-the-loop domain expert.
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id, parent_run_id=None, tags=None, **kwargs) -> None:
        if parent_run_id is not None:
            return  # only act on the top-level agent run, not internal graph nodes

        self.metrics["runs"] += 1
        client = _get_langsmith_client()

        # -- run-level latency & cost monitoring (diff against the start snapshot) --
        snap = self._run_starts.pop(str(run_id), None)
        if snap is not None and client is not None:
            run_latency = time.time() - snap["t"]
            comment = json.dumps({
                "run_latency_sec": round(run_latency, 3),
                "total_tokens": self.metrics["total_tokens"] - snap["tokens"],
                "cost_usd": round(self.metrics["cost_usd"] - snap["cost"], 6),
                "llm_calls": self.metrics["llm_calls"] - snap["llm_calls"],
            })
            try:
                client.create_feedback(run_id=run_id, key="run_latency_sec", score=round(run_latency, 3), comment=comment)
                client.create_feedback(run_id=run_id, key="run_cost_usd", score=round(self.metrics["cost_usd"] - snap["cost"], 6), comment=comment)
            except Exception as e:
                print(f"⚠️ Failed to push run latency/cost feedback to LangSmith: {e}")

        # -- faithfulness / relevance scoring on the final answer --
        if client is None:
            return  # avoid spending judge-LLM calls when there's nowhere to report scores
        if not isinstance(outputs, dict) or "messages" not in outputs:
            return

        question, context, answer = extract_question_context_answer(outputs["messages"])
        if not answer:
            return

        judge_llm = self._judge_llm or init_llm()
        faithfulness_score, faithfulness_rationale = score_faithfulness(context, answer, judge_llm)
        relevance_score, relevance_rationale = score_relevance(question, context, answer, judge_llm)
        answer_rel_score, answer_rel_rationale = score_answer_relevance(question, answer, judge_llm)

        try:
            client.create_feedback(run_id=run_id, key="faithfulness", score=faithfulness_score, comment=faithfulness_rationale)
            client.create_feedback(run_id=run_id, key="relevance", score=relevance_score, comment=relevance_rationale)
            client.create_feedback(run_id=run_id, key="answer_relevance", score=answer_rel_score, comment=answer_rel_rationale)
        except Exception as e:
            print(f"⚠️ Failed to push faithfulness/relevance feedback to LangSmith: {e}")


__all__ = [
    "init_langsmith_tracing",
    "AgenticObservabilityHandler",
    "score_faithfulness",
    "score_relevance",
    "score_answer_relevance",
    "extract_question_context_answer",
    # cost / token accounting
    "MODEL_PRICING",
    "lookup_pricing",
    "estimate_cost_usd",
]
