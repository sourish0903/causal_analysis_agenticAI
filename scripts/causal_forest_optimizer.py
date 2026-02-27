"""
Causal Forest Optimizer Module

Contains functions for estimating heterogeneous treatment effects using causal forests
and computing optimal lever combinations for achieving sales targets.
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

try:
    from econml.dml import CausalForestDML
except Exception as e:
    CausalForestDML = None
    _econml_import_error = e

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
from scipy.optimize import minimize


def estimate_changes_with_causal_forest(store, dept, levers_all=True, levers=None, data=None, confounders=None,
                                       tree_max_depth=3, min_samples_leaf=20,
                                       random_state=42, cf_params=None, top_n=2, max_lever_combo=2):
    """Estimate per-lever AND multi-lever required changes using a causal forest + interpretable tree on ITEs.
    
    Generates solutions for:
    1. Single levers (conditional on other levers)
    2. 2-lever combinations (joint effects)
    3. 3-lever combinations (triple interaction effects)

    Args:
      store, dept: unit identifiers
      levers_all: if True, auto-detect all markdown levers
      levers: list of lever base names (e.g., ['MarkDown1','MarkDown2']) if levers_all=False
      data: DataFrame (defaults to df_processed / store_dept_data)
      confounders: list of columns to use as controls/split features
      tree_max_depth/min_samples_leaf: parameters for small tree used to summarize ITEs
      cf_params: dict of params passed to CausalForestDML
      top_n: number of candidate solutions to return
      max_lever_combo: maximum number of levers in a combination (1-3)

    Returns:
      dict mapping lever/combo -> results dict with cf_model, tree, leaf summaries, etc.
      and list of lever names
    """
    if CausalForestDML is None:
        raise ImportError(f"econml is not available: {_econml_import_error}\nInstall with `pip install econml` and restart kernel.")

    rng = random.Random(random_state)

    if data is None:
        raise ValueError("data parameter is required")

    if confounders is None:
        confounders = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Month']

    subset = data[(data['Store'] == store) & (data['Dept'] == dept)].copy()
    if subset.empty:
        raise ValueError(f"No data found for Store={store}, Dept={dept}")

    subset = subset.reset_index(drop=True)
    
    if levers_all:
        levers = [col for col in subset.columns if col.startswith('MarkDown') or col.endswith('_applied')]
    else:
        levers = levers or []
    
    # Map levers to actual columns
    lever_map = {}
    for lv in levers:
        if lv in subset.columns:
            lever_map[lv] = {'col': lv, 'kind': 'continuous'}
        elif f"{lv}_applied" in subset.columns:
            lever_map[lv] = {'col': f"{lv}_applied", 'kind': 'binary'}
        else:
            print(f"⚠️ Lever {lv} not found in data; skipping")

    results = {}
    
    # Generate lever combinations: single (1), pairs (2), triples (3)
    max_lever_combo = min(max_lever_combo, 3)
    all_combos = []
    combos_by_size = {1: [], 2: [], 3: []}
    
    for combo_size in range(1, min(max_lever_combo + 1, len(lever_map) + 1)):
        size_combos = list(combinations(lever_map.keys(), combo_size))
        all_combos.extend(size_combos)
        combos_by_size[combo_size] = size_combos
    
    print(f"\n📊 Analyzing {len(all_combos)} lever combinations:")
    print(f"   - Single-lever: {len(combos_by_size[1])}")
    if 2 in combos_by_size:
        print(f"   - 2-lever combos: {len(combos_by_size[2])}")
    if 3 in combos_by_size:
        print(f"   - 3-lever combos: {len(combos_by_size[3])}")

    for combo in all_combos:
        combo_key = '+'.join(combo)
        
        # For single lever: use other levers as covariates
        # For multi-lever: estimate each lever's individual effect to enable optimization
        if len(combo) == 1:
            lv = combo[0]
            if lv not in lever_map:
                continue
            
            col = lever_map[lv]['col']
            kind = lever_map[lv]['kind']

            # Build X: confounders + OTHER levers (to capture interactions)
            X_cols = [c for c in confounders if c in subset.columns]
            
            # Add other levers as covariates
            for other_lv, info in lever_map.items():
                if other_lv != lv:
                    other_col = info['col']
                    if other_col in subset.columns:
                        X_cols.append(other_col)
            
            if len(X_cols) == 0:
                raise ValueError("No covariate columns found in data for causal forest")

            X = subset[X_cols].fillna(0).values
            T = subset[col].fillna(0).values
            Y = subset['Weekly_Sales'].values
            
            combo_kind = kind
            lever_effects = None  # Single lever, no need for individual effects
            
        else:
            # Multi-lever combination: estimate individual effects for each lever
            # This enables optimization to allocate changes optimally
            X_cols = [c for c in confounders if c in subset.columns]
            
            if len(X_cols) == 0:
                raise ValueError("No covariate columns found in data for causal forest")
            
            X = subset[X_cols].fillna(0).values
            Y = subset['Weekly_Sales'].values
            
            # Estimate effect for EACH lever in the combination
            lever_effects = {}
            combo_kinds = []
            
            for lv in combo:
                col = lever_map[lv]['col']
                kind = lever_map[lv]['kind']
                combo_kinds.append(kind)
                
                T_lv = subset[col].fillna(0).values
                
                # Fit causal forest for this lever
                model_y = RandomForestRegressor(n_estimators=100, random_state=random_state)
                model_t = RandomForestRegressor(n_estimators=100, random_state=random_state+1)
                
                cf_lv = CausalForestDML(
                    model_t=model_t, model_y=model_y,
                    discrete_treatment=(kind == 'binary'),
                    random_state=random_state,
                    n_estimators=100
                )
                
                try:
                    cf_lv.fit(Y, T_lv, X=X)
                    ite_lv = cf_lv.effect(X)
                    lever_effects[lv] = {'ite': ite_lv, 'kind': kind, 'col': col}
                except Exception as e:
                    print(f"⚠️ Failed to estimate effect for {lv} in combo {combo_key}: {e}")
                    lever_effects[lv] = None
            
            # Use average of individual effects as the combo effect for tree fitting
            valid_ites = [v['ite'] for v in lever_effects.values() if v is not None]
            if not valid_ites:
                print(f"⚠️ No valid lever effects for combo {combo_key}; skipping")
                continue
            
            ite = np.mean(valid_ites, axis=0)
            T = None  # No single treatment for multi-lever
            combo_kind = 'continuous' if 'continuous' in combo_kinds else 'binary'

        # Configure CausalForestDML params for single lever
        if len(combo) == 1:
            if cf_params is None:
                cf_params = {}
            
            model_y = RandomForestRegressor(n_estimators=100, random_state=random_state)
            model_t = RandomForestRegressor(n_estimators=100, random_state=random_state+1)

            cf = CausalForestDML(model_t=model_t, model_y=model_y,
                                 discrete_treatment=(combo_kind == 'binary'),
                                 random_state=random_state,
                                 n_estimators=100, **cf_params)

            print(f"\n🔬 Estimating effects for {combo_key} (kind={combo_kind}, {len(combo)}-lever) with {len(X_cols)} covariates...")

            try:
                cf.fit(Y, T, X=X)
            except Exception as e:
                print(f"⚠️ CausalForestDML failed for {combo_key}: {e}")
                results[combo_key] = None
                continue

            # Predict ITEs for each observation
            try:
                ite = cf.effect(X)
            except Exception as e:
                print(f"⚠️ Unable to get effect() from causal forest for {combo_key}: {e}")
                results[combo_key] = None
                continue
        else:
            print(f"\n🔬 Estimated individual effects for {len(combo)}-lever combo {combo_key} with {len(X_cols)} covariates")
            cf = None  # Multi-lever doesn't have a single causal forest

        # Fit small decision tree on ITEs to create interpretable leaves
        tree = DecisionTreeRegressor(max_depth=tree_max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
        try:
            tree.fit(X, ite)
            leaf_ids = tree.apply(X)
        except Exception as e:
            print(f"⚠️ Tree fitting on ITEs failed for {combo_key}: {e}")
            leaf_ids = np.zeros(len(subset), dtype=int)
            tree = None

        subset_combo = subset.assign(_leaf_id=leaf_ids)

        # Leaf summaries - store individual lever effects per leaf for optimization
        leaf_summary = []
        leaf_ite_map = {}
        leaf_lever_effects_map = {}
        
        for leaf, block in subset_combo.groupby('_leaf_id'):
            n = len(block)
            baseline = float(block['Weekly_Sales'].mean())
            mean_ite = float(ite[block.index].mean())
            
            # For multi-lever: store mean effect of each individual lever in this leaf
            if lever_effects:
                leaf_lever_effects = {}
                for lv, effect_data in lever_effects.items():
                    if effect_data is not None:
                        leaf_lever_effects[lv] = {
                            'mean_ite': float(effect_data['ite'][block.index].mean()),
                            'kind': effect_data['kind']
                        }
                leaf_lever_effects_map[int(leaf)] = leaf_lever_effects
            
            leaf_summary.append({
                'leaf': int(leaf), 
                'n': int(n), 
                'baseline_sales': baseline, 
                'mean_ite': mean_ite, 
                'leaf_share': float(n / len(subset_combo))
            })
            leaf_ite_map[int(leaf)] = mean_ite

        leaf_df = pd.DataFrame(leaf_summary).sort_values('n', ascending=False).reset_index(drop=True)

        results[combo_key] = {
            'cf_model': cf,
            'tree_model': tree,
            'leaf_df': leaf_df,
            'leaf_ite_map': leaf_ite_map,
            'leaf_lever_effects': leaf_lever_effects_map if lever_effects else None,
            'treatment_kind': combo_kind,
            'treatment_combo': combo,
            'num_levers': len(combo),
            'covariate_cols': X_cols
        }
        
        print(f"✓ Completed: {len(leaf_df)} leaves found with mean ITE range [{leaf_df['mean_ite'].min():.2f}, {leaf_df['mean_ite'].max():.2f}]")

    print(f"\n✅ Successfully analyzed {len([r for r in results.values() if r is not None])}/{len(all_combos)} combinations")
    return results, levers


def compute_required_changes_from_causal_forest(results, target_pct, levers, top_n=15, random_state=42, 
                                                ensure_multi_lever=False):
    """Given results from estimate_changes_with_causal_forest, compute required deltas per leaf
    for BOTH single-lever and multi-lever combinations using OPTIMIZATION for multi-lever.
    Returns a combined DataFrame of candidate solutions ranked by feasibility and magnitude.
    
    For multi-lever solutions: uses scipy.optimize to find the optimal allocation of changes
    across levers that minimizes total intervention magnitude while achieving target.
    
    Args:
        results: dict from estimate_changes_with_causal_forest
        target_pct: target percentage change in outcome
        levers: list of lever names
        top_n: total number of solutions to return
        random_state: random seed
        ensure_multi_lever: if True, guarantees multi-lever solutions in output
    """
    rng = random.Random(random_state)
    rows = []
    
    # Track counts for diagnostics
    diagnostic_counts = {'single': 0, 'multi_2': 0, 'multi_3': 0, 'failed': 0}

    for combo_key, res in results.items():
        if not res:
            diagnostic_counts['failed'] += 1
            continue
        
        leaf_df = res['leaf_df']
        leaf_ite_map = res['leaf_ite_map']
        kind = res['treatment_kind']
        combo = res['treatment_combo']
        num_levers = res['num_levers']
        leaf_lever_effects = res.get('leaf_lever_effects')

        for _, row in leaf_df.iterrows():
            leaf = int(row['leaf'])
            n = int(row['n'])
            baseline = float(row['baseline_sales'])
            mean_ite = float(leaf_ite_map.get(leaf, 0.0))
            
            if abs(mean_ite) < 1e-8:
                continue
            
            target_change = baseline * (target_pct / 100.0)
            
            if num_levers == 1:
                # Single lever: delta is straightforward
                delta = target_change / mean_ite
                unit = 'amount' if kind == 'continuous' else 'proportion'
                feasible = True
                if unit == 'proportion':
                    feasible = (-1.0 <= delta <= 1.0)
                
                desc = f"Leaf {leaf} ({n} obs): {combo_key} by {delta:.4f} {unit}"
                delta_tuple = (delta,)
                score = abs(delta)
                diagnostic_counts['single'] += 1
                
            else:
                # Multi-lever: use OPTIMIZATION to find best allocation
                if not leaf_lever_effects or leaf not in leaf_lever_effects:
                    continue
                
                lever_effects_leaf = leaf_lever_effects[leaf]
                
                # Extract individual lever effects
                lever_names = list(lever_effects_leaf.keys())
                lever_ites = np.array([lever_effects_leaf[lv]['mean_ite'] for lv in lever_names])
                lever_kinds = [lever_effects_leaf[lv]['kind'] for lv in lever_names]
                
                if np.all(np.abs(lever_ites) < 1e-8):
                    continue
                
                # Optimization objective: minimize L2 norm of changes
                def objective(deltas):
                    return np.sum(deltas**2)
                
                # Constraint: sum of (delta_i * ite_i) = target_change
                def constraint_eq(deltas):
                    return np.dot(deltas, lever_ites) - target_change
                
                # Initial guess: equal split
                x0 = np.ones(num_levers) * (target_change / np.sum(lever_ites)) if np.sum(lever_ites) != 0 else np.ones(num_levers)
                
                # Bounds: reasonable limits for each lever type
                bounds = []
                for lv_kind in lever_kinds:
                    if lv_kind == 'binary':
                        bounds.append((-1.0, 1.0))  # proportion change
                    else:
                        bounds.append((None, None))  # continuous, no strict bounds
                
                constraints = {'type': 'eq', 'fun': constraint_eq}
                
                try:
                    result = minimize(
                        objective, 
                        x0, 
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000}
                    )
                    
                    if result.success:
                        optimal_deltas = result.x
                        feasible = True
                        
                        # Check feasibility for binary levers
                        for i, lv_kind in enumerate(lever_kinds):
                            if lv_kind == 'binary' and not (-1.0 <= optimal_deltas[i] <= 1.0):
                                feasible = False
                                break
                    else:
                        # Fallback to equal split if optimization fails
                        optimal_deltas = x0
                        feasible = False
                        
                except Exception as e:
                    print(f"⚠️ Optimization failed for {combo_key}, leaf {leaf}: {e}")
                    optimal_deltas = x0
                    feasible = False
                
                # Format description
                delta_strs = [f"{lever_names[i]}={optimal_deltas[i]:.4f}" for i in range(num_levers)]
                unit = f'{num_levers}-lever optimized'
                delta_tuple = tuple(optimal_deltas)
                desc = f"Leaf {leaf} ({n} obs): {combo_key} → {', '.join(delta_strs)} [{unit}]"
                score = np.sqrt(np.sum(optimal_deltas**2))  # L2 norm as score
                
                if num_levers == 2:
                    diagnostic_counts['multi_2'] += 1
                elif num_levers == 3:
                    diagnostic_counts['multi_3'] += 1

            rows.append({
                'combo': combo_key,
                'num_levers': num_levers,
                'leaf': leaf,
                'n': n,
                'required_change': delta_tuple,
                'unit': unit,
                'description': desc,
                'feasible': feasible,
                'score': score
            })

    print(f"\n📊 Generated candidate solutions:")
    print(f"   - Single-lever: {diagnostic_counts['single']}")
    print(f"   - 2-lever: {diagnostic_counts['multi_2']}")
    print(f"   - 3-lever: {diagnostic_counts['multi_3']}")
    print(f"   - Failed: {diagnostic_counts['failed']}")

    # Separate by lever count and feasibility
    feasible = [r for r in rows if r['feasible']]
    infeasible = [r for r in rows if not r['feasible']]
    
    feasible.sort(key=lambda x: (x['num_levers'], x['score'], -x['n']))
    infeasible.sort(key=lambda x: (x['num_levers'], x['score'], -x['n']))
    
    # If ensure_multi_lever is True, guarantee diversity in output
    if ensure_multi_lever and feasible:
        # Separate by num_levers
        by_lever_count = {1: [], 2: [], 3: []}
        for r in feasible:
            by_lever_count[r['num_levers']].append(r)
        
        # Select proportionally: aim for mix of single, 2-lever, 3-lever
        picked = []
        n_per_type = max(1, top_n // 3)  # At least 1 of each type if available
        
        # Get best from each category
        for lever_count in [1, 2, 3]:
            if by_lever_count[lever_count]:
                candidates = by_lever_count[lever_count][:max(5, n_per_type*2)]
                n_pick = min(n_per_type, len(candidates))
                if n_pick > 0:
                    picked.extend(rng.sample(candidates, n_pick))
        
        # Fill remaining slots with best overall
        remaining = top_n - len(picked)
        if remaining > 0:
            all_remaining = [r for r in feasible if r not in picked]
            if all_remaining:
                n_pick = min(remaining, len(all_remaining))
                picked.extend(rng.sample(all_remaining[:max(10, remaining*2)], n_pick))
    else:
        # Original random sampling from top candidates
        if feasible:
            pool = feasible[:max(20, len(feasible))]
            picked = rng.sample(pool, min(top_n, len(pool)))
        else:
            pool = infeasible[:max(20, len(infeasible))]
            picked = rng.sample(pool, min(top_n, len(pool))) if pool else []

    out = []
    for sol in picked:
        out.append({
            'combo': sol['combo'],
            'num_levers': sol['num_levers'],
            'leaf': sol['leaf'],
            'n': sol['n'],
            'required_change': sol['required_change'],
            'unit': sol['unit'],
            'description': sol['description']
        })

    result_df = pd.DataFrame(out).sort_values(['num_levers', 'n'], ascending=[True, False])
    
    print(f"\n🎯 Returning {len(result_df)} diverse candidate solutions:")
    if not result_df.empty:
        print(f"   ✓ Single-lever solutions: {len(result_df[result_df['num_levers']==1])}")
        print(f"   ✓ 2-lever solutions: {len(result_df[result_df['num_levers']==2])}")
        print(f"   ✓ 3-lever solutions: {len(result_df[result_df['num_levers']==3])}")
        print("\n📋 Top solutions by lever count:")
        for i, r in result_df.head(15).iterrows():
            lever_emoji = "🎯" if r['num_levers'] == 1 else "🎯🎯" if r['num_levers'] == 2 else "🎯🎯🎯"
            print(f"  {lever_emoji} {r['description']}")
    
    return result_df
