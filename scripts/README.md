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

    - Structural model (display):

      $$
      Y = g(X) + \tau(X)\cdot T + \varepsilon,
      $$

      where $Y$ is the outcome (Weekly\_Sales), $T$ is the treatment (a lever), $X$ are covariates/confounders, $g(\cdot)$ is the baseline response surface, and $\tau(X)$ is the heterogeneous treatment effect (HTE).

    - Double Machine Learning (DML) orthogonalization (display):

      Estimate the nuisance functions
      $$\hat m(X) \approx \mathbb{E}[Y\mid X], \quad \hat p(X) \approx \mathbb{E}[T\mid X],$$
      form residuals
      $$\tilde Y = Y - \hat m(X), \quad \tilde T = T - \hat p(X).$$

      Then for each $X$ solve a local regression of $\tilde Y$ on $\tilde T$ to recover the local slope (the HTE). Formally one may write the target as the minimizer
      $$
      \tau^*(X) = \arg\min_{f} \; \mathbb{E}\big[(\tilde Y - f(X)\,\tilde T)^2 \mid X\big],
      $$
      whose closed-form population solution (when moments exist) is
      $$
      \tau^*(X) = \frac{\mathbb{E}[\tilde Y\,\tilde T \mid X]}{\mathbb{E}[\tilde T^2 \mid X]}.
      $$

      DML uses cross-fitting to obtain orthogonal (debiased) estimates of these nuisances so the downstream estimator for $\tau(X)$ is less sensitive to regularization error in nuisance learners.

    - Causal forests (intuition):

      `CausalForestDML` fits an ensemble of trees to estimate the mapping $X\mapsto\tau(X)$. Each tree partitions the covariate space and produces local treatment effect estimates; the forest aggregates these to produce observation-level ITEs $\hat\tau(X_i)$ and provides variance/uncertainty estimates over ITEs.

- `causal_agentic_ai.py`
  - Purpose: High-level entry point and agent builder that ties LLMs (LangChain / LangGraph) to the analytic tools. Exposes helper functions and a `build_agent` function that composes an agent workflow and binds the analytic tool functions as callable tools for the LLM.
  - Role: Coordinates model initialization, tool wiring, and agent execution. Not responsible for core econometric estimation logic (delegates to `causal_forest_optimizer.py` and `tools.py`).

- `tools.py`
  - Purpose: Lightweight toolkit and glue code used by both the Streamlit UI and the agent. Contains helpers for state management, data normalization, parsing, and higher-level wrappers around the causal forest routines (e.g., `find_optimal_levers`, `analyze_whatif_scenario`, `analyze_variable_impact`).
  - Role: Adapt raw data into the shapes required by the causal estimators and provide curated outputs for UI/agent consumption.

Notes

- Reproducibility: `econml` (and its dependencies) must be installed in the environment to use `CausalForestDML`.
- References: The DML approach follows the Chernozhukov et al. (2018) double/debiased machine learning framework and the idea of causal forests for HTEs (Wager & Athey). For more details consult the `econml` documentation and the cited papers.
