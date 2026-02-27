---
noteId: "31cf541013b711f1809c9171d8b1300c"
tags: []

---
# Scripts Overview

This folder contains the main analysis and agent-building scripts used by the repository.

Files

- `causal_forest_optimizer.py`
  - Purpose: Estimate heterogeneous treatment effects (HTEs) for marketing levers and compute candidate interventions (single and multi-lever) to achieve target outcome changes.
  - Key idea: Uses econml's CausalForestDML to estimate τ(x), an observation-specific treatment effect, and fits a small decision tree on the estimated ITEs to produce interpretable population segments (leaves) for optimization.
  - Dependency: `econml` (imported as `from econml.dml import CausalForestDML`).
  - Brief mathematical formulation:
    - Structural model: Y = g(X) + τ(X)·T + ε, where Y is outcome (Weekly_Sales), T is treatment (lever), X are covariates/confounders, and τ(X) is the heterogeneous treatment effect.
    - Double machine learning (DML) orthogonalization: estimate nuisance functions m(X)=E[Y|X] and p(X)=E[T|X], form residuals \tilde Y = Y - m̂(X), \tilde T = T - p̂(X), then estimate τ(X) by fitting a flexible model (here a causal forest) to learn the mapping X ↦ E[\tilde Y | X, \tilde T]/E[\tilde T^2 | X] (intuitively the local slope of Y on T after removing confounding).
    - CausalForestDML implements this procedure with ensemble trees to produce observation-level ITEs τ̂_i = τ̂(X_i).

- `causal_agentic_ai.py`
  - Purpose: High-level entry point and agent builder that ties LLMs (LangChain / LangGraph) to the analytic tools. Exposes helper functions and a `build_agent` function that composes an agent workflow and binds the analytic tool functions as callable tools for the LLM.
  - Role: Coordinates model initialization, tool wiring, and agent execution. Not responsible for core econometric estimation logic (delegates to `causal_forest_optimizer.py` and `tools.py`).

- `tools.py`
  - Purpose: Lightweight toolkit and glue code used by both the Streamlit UI and the agent. Contains helpers for state management, data normalization, parsing, and higher-level wrappers around the causal forest routines (e.g., `find_optimal_levers`, `analyze_whatif_scenario`, `analyze_variable_impact`).
  - Role: Adapt raw data into the shapes required by the causal estimators and provide curated outputs for UI/agent consumption.

Notes

- Reproducibility: `econml` (and its dependencies) must be installed in the environment to use `CausalForestDML`.
- References: The DML approach follows the Chernozhukov et al. (2018) double/debiased machine learning framework and the idea of causal forests for HTEs (Wager & Athey). For more details consult the `econml` documentation and the cited papers.
