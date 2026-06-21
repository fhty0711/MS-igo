# MGIGO Scenario Planner

This repository is the clean MGIGO scenario-planning subset extracted from the
larger IOC_AGV workspace. It keeps only the active autonomous-driving planning
path:

```text
run_mgigo_scenario.py
  -> scenarios/
  -> costs/
  -> backends/
  -> planner.py
  -> igo/MPC_G_MS.py
  -> scenario_runtime.py
  -> viz_utils.py
```

## What Is Included

- Generic scenario interface: `scenarios/spec.py`
- Registered highway merge scenario: `scenarios/highway_merge.py`
- Constraint-DSL cost design: `costs/constraint_dsl.py`
- Active highway cost profile: `costs/highway_merge.py`
- Generic MGIGO planner: `planner.py`
- Multi-agent MGIGO solver: `igo/MPC_G_MS.py`
- Closed-loop runtime and visualization: `scenario_runtime.py`, `viz_utils.py`

Old IOC, MRS, presentation, generated videos, and legacy scripts are not
included in this clean repository.

## Run

Use WSL2 + CUDA when available:

```bash
cd igo
uv sync
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py highway_merge highway_merge
```

Outputs are written to:

```text
figures/mgigo_highway_snapshot.png
figures/mgigo_highway.mp4
```

CPU can run the code but is much slower.

## Add A New Scenario

1. Add `scenarios/<new_scenario>.py` with a `make_scenario() -> ScenarioSpec`.
2. Register it in `scenarios/registry.py`.
3. Add a matching constraint-DSL cost file in `costs/`.
4. Register the cost in `costs/registry.py`.
5. Run:

```bash
uv run python run_mgigo_scenario.py <new_scenario> <new_cost>
```

See `ARCHITECTURE.md` for the full call chain.
