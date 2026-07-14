# Unified Scene Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor visualization so all scenarios share one layer-based drawing core while preserving current rendered output.

**Architecture:** Add a standalone `visualization.scene_renderer` module with plain dataclasses for map, line, vehicle, arrow, circle, text, and legend layers. Keep `viz_utils.render_agents_panel(...)` as the external entrypoint, but make both ordinary road scenes and signalized-intersection scenes build render specs consumed by the same renderer. Keep scenario-specific code as adapters that describe layers; the renderer does not branch on scenario names.

**Tech Stack:** Python dataclasses, NumPy, Matplotlib Agg tests, existing `ScenarioSpec` and visualization tests.

---

### Task 1: Add the Generic Render Spec Contract

**Files:**
- Create: `visualization/scene_renderer.py`
- Modify: `visualization/__init__.py`
- Test: `tests/test_unified_scene_renderer.py`

- [ ] **Step 1: Write the failing test**

Add `tests/test_unified_scene_renderer.py` with tests that import `SceneRenderSpec`, create a simple scene with one rectangle and one line, call `render_scene(ax, spec)`, and assert patch/line gids exist.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests\test_unified_scene_renderer.py`
Expected: fails with `ModuleNotFoundError` or missing `SceneRenderSpec`.

- [ ] **Step 3: Implement minimal renderer**

Create dataclasses for `SceneRenderSpec`, `RectLayer`, `LineLayer`, `CircleLayer`, `ArrowLayer`, `TextLayer`, `VehicleLayer`, and `LegendLayer`. Implement `render_scene(ax, spec)` and `draw_vehicle_footprint(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests\test_unified_scene_renderer.py`
Expected: prints `unified scene renderer tests ok`.

### Task 2: Route Generic Highway/Borrow Road Drawing Through the Unified Renderer

**Files:**
- Modify: `viz_utils.py`
- Test: `tests/test_generic_renderer_smoke.py`
- Test: `tests/test_unified_scene_renderer.py`

- [ ] **Step 1: Extend tests with pixel parity**

Add a test helper that renders `highway_merge` and `borrow_overtake_safe` with `render_agents_panel(...)` and asserts the output is nonblank. The existing smoke test already covers this; keep it as the regression gate.

- [ ] **Step 2: Refactor `_draw_rect` and `_draw_road`**

Make `_draw_rect(...)` call `visualization.scene_renderer.draw_vehicle_footprint(...)`. Make `_draw_road(...)` construct a `SceneRenderSpec` road/lane line layer set and call `render_scene(...)`.

- [ ] **Step 3: Run generic renderer tests**

Run: `python tests\test_generic_renderer_smoke.py`
Expected: all generic highway/borrow scenarios still render and do not import `viz_signalized`.

### Task 3: Route Signalized Map Drawing Through the Unified Renderer

**Files:**
- Modify: `visualization/signalized_renderer.py`
- Modify: `viz_signalized.py`
- Test: `tests/test_signalized_renderer_standalone.py`
- Test: `tests/test_signalized_intersection_helpers.py`

- [ ] **Step 1: Add signalized layer contract assertions**

Keep existing gid assertions for `lane_polygon`, `crosswalk`, `conflict_box`, and `traffic_signal_red_active`.

- [ ] **Step 2: Refactor signalized map helpers**

Change `draw_signalized_scene(...)` to construct generic layers and call `render_scene(...)` for the semantic map, lane dividers, stop line, crosswalks, arrows, signal post, and signal lamps. Keep cross-traffic and metric helpers compatible.

- [ ] **Step 3: Run signalized tests**

Run: `python tests\test_signalized_renderer_standalone.py`
Run: `python tests\test_signalized_intersection_helpers.py`
Expected: both pass.

### Task 4: Pixel/Smoke Verification Across Scenarios

**Files:**
- Test-only generated outputs under `figures/_smoke_*.png`

- [ ] **Step 1: Generate non-signalized smoke images**

Run a Matplotlib Agg script rendering `highway_merge`, `borrow_overtake_safe`, `borrow_overtake_blocked`, and `borrow_overtake_critical`.

- [ ] **Step 2: Read generated images**

Read PNGs back with `matplotlib.image.imread` and assert nonzero standard deviation.

- [ ] **Step 3: Run compile and test suite subset**

Run:
`python -m py_compile visualization\scene_renderer.py visualization\signalized_renderer.py viz_signalized.py viz_utils.py tests\test_unified_scene_renderer.py tests\test_generic_renderer_smoke.py tests\test_signalized_renderer_standalone.py tests\test_signalized_intersection_helpers.py`

Run:
`python tests\test_unified_scene_renderer.py`
`python tests\test_generic_renderer_smoke.py`
`python tests\test_signalized_renderer_standalone.py`
`python tests\test_signalized_intersection_helpers.py`

Expected: all commands exit 0.

---

Self-review:
- The plan keeps all main code changes under `igo/`.
- It preserves the public `render_agents_panel(...)` entrypoint.
- It removes drawing primitive duplication by routing ordinary and signalized scenes through `visualization.scene_renderer`.
- It uses existing smoke/semantic tests plus compile checks as verification.
