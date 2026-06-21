# MGIGO Scenario Architecture

This project now uses the generic scenario architecture for MGIGO experiments.
Legacy fixed-highway scripts are archived under `archive/legacy/`.

## Active Entry Point

```bash
uv run python run_mgigo_scenario.py
uv run python run_mgigo_scenario.py highway_merge highway_merge
```

The first argument selects a registered scenario. The second argument selects a
registered cost profile. If the cost profile is omitted, the scenario's default
profile is used. Active cost profiles should use the constraint DSL. Legacy
hand-written costs are archived and are not part of the active registry.

## Call Chain

```text
run_mgigo_scenario.py
  -> scenarios.get_scenario(name)
  -> costs.get_cost_functions(profile)
  -> backends.get_backend(scenario.backend)
  -> GenericScenarioBackend.step()
  -> planner.plan()
  -> igo.MPC_G_MS.mmog_igo_rne_blocks_solver()
  -> costs.<profile>.<agent>_cost()
  -> costs.common.dense_rollout_from_decisions()
  -> scenario_runtime.advance_one_macro_step()
  -> viz_utils.render_agents_panel()
```

## Core Modules

- `scenarios/spec.py`: dataclasses for `AgentSpec`, `DecisionSpec`,
  `BlockSpec`, `SolverSpec`, road geometry, vehicle geometry, and
  `ScenarioSpec` validation.
- `scenarios/<name>.py`: one scenario per file. Keep scenario-specific vehicle
  count, initial states, references, decisions, blocks, and drawing labels here.
- `scenarios/registry.py`: imports scenario factories and registers scenario
  names.
- `costs/constraint_dsl.py`: Constran-style builders for objective functions
  and directly evaluated constraints (`g(x, ctx) <= 0`) including
  deterministic, chance, robust, and DRO forms.
- `costs/<profile>.py`: one cost profile per file. The active design direction
  is to express each agent's objective and constraints through the DSL and
  return one scalar black-box objective per agent.
- `costs/registry.py`: registers independently selectable cost profiles.
- `decision_layout.py`: decodes sampled MGIGO blocks into named decision
  sequences and encodes/warms them back into block arrays.
- `planner.py`: generic MGIGO planning step. It builds context, calls the
  solver, selects components, decodes decisions, rolls out trajectories, and
  returns warm-start data.
- `scenario_runtime.py`: NumPy closed-loop execution and prediction helpers for
  decoded scenario decisions.
- `costs/common.py`: JAX rollout and shared cost helpers used inside black-box
  objective evaluation.
- `backends/generic_scenario.py`: scenario-independent execution history,
  progress logging, and rendering adapter.
- `viz_utils.py`: generic multi-agent road rendering.

## Adding a New Experiment

1. Add `scenarios/<new_scenario>.py` with a `make_scenario()` factory.
2. Register it in `scenarios/registry.py`.
3. Add one or more Constran-style cost profiles in `costs/`.
4. Register each cost profile in `costs/registry.py`.
5. Run:

```bash
uv run python run_mgigo_scenario.py <new_scenario> <cost_profile>
```

The active path should not import from `archive/legacy/`.
