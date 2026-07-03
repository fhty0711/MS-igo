"""
planner.py - generic scenario-aware MGIGO planner.

提供：
  make_fitness_fn                按 cost profile 生成博弈代价分发器
  plan                           博弈 MGIGO 规划一步
"""

from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, jit

from igo.MPC_G_MS import mmog_igo_rne_blocks_solver

from config import (
    K_COMP, B_SAMP, B_ELITE, T_OPT, ALPHA_T, M_INNER, T_RESET,
    EXEC_MODE, WARM_START_NOISE_STD,
    INIT_MU_NOISE_STD,
    CTRL_SMOOTH_ALPHA,
)
from costs import get_cost_functions
from component_selection import select_components_by_cost, select_components_by_weight
from decision_layout import BlockDecoder
from scenario_runtime import (
    advance_one_macro_step,
    dense_rollout_np,
    prediction_trajs,
)
from scenarios import get_scenario
from trajectory.frenet_bspline import FrenetBSplineTrajectory
from trajectory.reference_path import StraightReference
from trajectory.rollout import bspline_context_from_state
from trajectory.warmstart import tangent_control_points


_DEFAULT_SCENARIO = get_scenario("highway_merge")
_DEFAULT_SOLVER_SPEC = _DEFAULT_SCENARIO.solver_spec

# ══════════════════════════════════════════════════════════════════════════════
#  博弈代价分发器
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def make_fitness_fn(cost_profile=None):
    """Build the jitted agent-cost dispatcher for a scenario cost profile.

    The MGIGO solver calls one function with ``agent_idx``. This dispatcher
    routes that call to the corresponding registered agent cost.
    """
    agent_costs = tuple(get_cost_functions(cost_profile))
    if not agent_costs:
        raise ValueError(f"Cost profile {cost_profile!r} has no agent cost functions")

    branches = tuple(
        (lambda cost_fn: (lambda s, c: cost_fn(s, c)))(cost_fn)
        for cost_fn in agent_costs
    )

    @jit
    def fitness_fn(agent_idx, joint_sample_flat, context_arr):
        return lax.switch(
            agent_idx,
            branches,
            joint_sample_flat,
            context_arr,
        )

    return fitness_fn


def _solver_spec_or_default(solver_spec):
    """Use the provided solver layout or the default highway layout."""
    return solver_spec if solver_spec is not None else _DEFAULT_SOLVER_SPEC


def _scenario_or_default(scenario):
    """Use the provided scenario or the registered highway default."""
    return scenario if scenario is not None else _DEFAULT_SCENARIO


def _shift_sequences(sequences, decoder=None):
    """Shift raw block sequences forward by one MPC index."""
    sequences = np.asarray(sequences, dtype=np.float32)
    if decoder is not None:
        return decoder.shift_blocks(sequences).astype(np.float32)
    return np.concatenate(
        [sequences[:, 1:], np.zeros((sequences.shape[0], 1), dtype=sequences.dtype)],
        axis=1,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  热启动（Algorithm 3 风格）
#
#  上一轮真正执行的 component mean 前移一拍，作为下一轮每个 block 的
#  component 0；其他 component 围绕它加噪声探索。由于 component 语义被
#  重新编号，L/v 每个 MPC 步重置为 identity/zeros，避免旧精度矩阵和旧权重
#  与新加噪 mu 失配。
# ══════════════════════════════════════════════════════════════════════════════

def _warm_start_mu(final_mu_np, best_ks, rng, solver_spec=None, best_seqs=None,
                   noise_std=WARM_START_NOISE_STD, decoder=None):
    """Shift the executed block sequences and seed the next GMM means.

    Component 0 is the shifted previous plan. Other components add Gaussian
    exploration noise around that shifted plan.
    """
    solver_spec = _solver_spec_or_default(solver_spec)
    n_blocks = solver_spec.n_blocks
    solver_width = max(tuple(solver_spec.block_dims))
    if best_seqs is None:
        best_seqs = np.stack([
            final_mu_np[block, best_ks[block]]
            for block in range(n_blocks)
        ])
    else:
        best_seqs = np.asarray(best_seqs, dtype=np.float32)
    shifted = _shift_sequences(best_seqs, decoder=decoder)
    noise = rng.normal(0, noise_std, size=(n_blocks, K_COMP, solver_width))
    base = np.tile(shifted[:, None, :], (1, K_COMP, 1))
    k_mask = (np.arange(K_COMP) == 0)[None, :, None]
    return np.where(k_mask, base, base + noise).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  采样（用于可视化轨迹云）
# ══════════════════════════════════════════════════════════════════════════════

def _sample_block_z(final_pi_np, final_mu_np, final_L_np, rng, solver_spec=None):
    """Sample one block vector per block from the final learned GMM."""
    solver_spec = _solver_spec_or_default(solver_spec)
    n_blocks = solver_spec.n_blocks
    solver_width = max(tuple(solver_spec.block_dims))
    z = np.zeros((n_blocks, solver_width), dtype=np.float64)
    for blk in range(n_blocks):
        k = rng.choice(K_COMP, p=final_pi_np[blk] / final_pi_np[blk].sum())
        S_k = final_L_np[blk, k] @ final_L_np[blk, k].T
        try:
            cov = np.linalg.inv(S_k + np.eye(solver_width) * 5e-4)
            cov = 0.5 * (cov + cov.T)
            if np.any(np.linalg.eigvalsh(cov) < 0):
                raise ValueError
            z[blk] = rng.multivariate_normal(final_mu_np[blk, k], cov)
        except Exception:
            z[blk] = final_mu_np[blk, k] + rng.standard_normal(solver_width) * 0.1
    return z


def _bspline_seed_blocks(scenario, current_states, solver_width):
    """Return B-spline control-point blocks consistent with the current state."""
    from pathlib import Path

    basis_path = (
        Path(__file__).resolve().parent
        / "trajectory"
        / "assets"
        / "bspline_basis.npz"
    )
    gen = FrenetBSplineTrajectory(basis_path, StraightReference())
    ctx = bspline_context_from_state(current_states[0])
    ctrl_s, ctrl_d = tangent_control_points(
        gen,
        s0=float(ctx["s0"]),
        s_dot0=float(ctx["s_dot0"]),
        d0=float(ctx["d0"]),
    )
    seed = np.zeros((scenario.solver_spec.n_blocks, solver_width), dtype=np.float32)
    seed[0, :len(ctrl_s)] = ctrl_s
    seed[1, :len(ctrl_d)] = ctrl_d
    return seed


# ══════════════════════════════════════════════════════════════════════════════
#  博弈规划器入口
# ══════════════════════════════════════════════════════════════════════════════

def plan(
    key,
    current_states,
    v_refs,
    warm=None,
    cost_profile=None,
    scenario=None,
    solver_spec=None,
    elapsed_time_s=0.0,
):
    """
    博弈 MGIGO 规划一步。

    参数
    ----
    key            : JAX PRNGKey
    current_states : (n_agents, state_dim) np.ndarray
    v_refs         : (n_agents,)
    warm           : (final_mu, exec_k, selected_blocks) | None
                      热启动状态：GMM 均值 + 上轮选中分量 + 平滑后的 block 序列
    cost_profile   : str  costs.registry 中注册的代价函数组

    返回
    ----
    dict:
      warm          : (final_mu_np, exec_k)  传给下一轮的 warm 参数
      control_sequences_by_block : dict      各 block 入选控制序列
      decision_sequences         : dict      命名 decision 序列
      dense_traj                 : ndarray   全轨迹稠密展开
      next_states                : ndarray   推进一个宏步后的状态
      warm          : tuple                   下一轮热启动
      trajectories_by_agent      : dict      每个 agent 的采样轨迹云
      best_block_ks              : ndarray   各 block 入选分量索引
    """
    scenario = _scenario_or_default(scenario)
    cost_profile = cost_profile or scenario.cost_profile
    decoder = BlockDecoder(scenario)
    solver_spec = _solver_spec_or_default(solver_spec or scenario.solver_spec)
    n_blocks = solver_spec.n_blocks
    m_agent = solver_spec.m_agent
    block_dims = tuple(solver_spec.block_dims)
    block_to_agent = tuple(solver_spec.block_to_agent)
    solver_width = max(block_dims)

    context_arr = jnp.concatenate([
        jnp.asarray(current_states.reshape(-1), dtype=jnp.float32),
        jnp.asarray(v_refs, dtype=jnp.float32),
        jnp.asarray(scenario.context_values, dtype=jnp.float32),
        jnp.asarray([elapsed_time_s], dtype=jnp.float32),
    ])
    fitness_fn = make_fitness_fn(cost_profile)

    # ── 初始化 / 热启动 ────────────────────────────────────────────────
    # Algorithm 3 执行上一轮选中 component mean；warm start 将
    # 该执行序列 shift-1 后放到下一轮 component 0，其余 component 加噪声。
    # component 被重新编号后，L/v 也重置，避免旧权重和旧精度矩阵绑定到
    # 已经改变语义的 component 上。
    prec_scale = 1.0
    L_identity = np.tile(
        (np.eye(solver_width, dtype=np.float32) * np.sqrt(prec_scale))[None, None],
        (n_blocks, K_COMP, 1, 1),
    )
    v_zeros = np.zeros((n_blocks, K_COMP - 1), dtype=np.float32)
    rng = np.random.default_rng(int(np.asarray(key[0])))
    if warm is None:
        mu0 = rng.normal(
            0.0, INIT_MU_NOISE_STD,
            size=(n_blocks, K_COMP, solver_width),
        ).astype(np.float32)
        for block_idx, component_values in enumerate(scenario.initial_component_means):
            if block_idx >= n_blocks:
                break
            for comp_idx, value in enumerate(component_values):
                if comp_idx >= K_COMP:
                    break
                value_arr = np.asarray(value, dtype=np.float32)
                if value_arr.ndim == 0:
                    mu0[block_idx, comp_idx, :block_dims[block_idx]] += float(value_arr)
                elif value_arr.shape == (block_dims[block_idx],):
                    mu0[block_idx, comp_idx, :block_dims[block_idx]] += value_arr
                else:
                    raise ValueError(
                        "initial_component_means entries must be scalars or "
                        f"vectors of length block_dim={block_dims[block_idx]}; "
                        f"got shape {value_arr.shape} for block {block_idx}, "
                        f"component {comp_idx}"
                    )
        L_inv0 = L_identity
        v0 = v_zeros
        prev_exec_k = None
        prev_selected_blocks = None
    else:
        prev_mu_np, last_exec_k = warm[:2]
        prev_selected_blocks = np.asarray(warm[2], dtype=np.float32)
        # Step 1: 用上轮实际执行的平滑序列 shift-1 → 新 component 0。
        # B-spline 决策是绝对控制点而非逐步控制序列，不能左移补零；
        # 每个 MPC 步从当前状态重新生成 tangent control points 作为先验。
        if scenario.trajectory_model == "frenet_bspline":
            shifted_seed = _bspline_seed_blocks(scenario, current_states, solver_width)
            noise = rng.normal(
                0,
                WARM_START_NOISE_STD,
                size=(n_blocks, K_COMP, solver_width),
            )
            base = np.tile(shifted_seed[:, None, :], (1, K_COMP, 1))
            k_mask = (np.arange(K_COMP) == 0)[None, :, None]
            mu0 = np.where(k_mask, base, base + noise).astype(np.float32)
        else:
            mu0 = _warm_start_mu(prev_mu_np, last_exec_k, rng,
                                 solver_spec=solver_spec,
                                 best_seqs=prev_selected_blocks,
                                 decoder=decoder)
        L_inv0 = L_identity
        v0 = v_zeros
        # Step 2: component 0 即上轮执行的延续，hysteresis 应向 0 bias
        prev_exec_k = np.zeros((n_blocks,), dtype=np.int32)

    # ── 调用 MGIGO 求解器 ──────────────────────────────────────────────
    final_mu, final_L, final_pi, _ = mmog_igo_rne_blocks_solver(
        key                = key,
        T                  = T_OPT,
        dt                 = ALPHA_T,
        N_blocks           = n_blocks,
        M_agent            = m_agent,
        K                  = K_COMP,
        B                  = B_SAMP,
        B0                 = B_ELITE,
        dims               = block_dims,
        T_0                = T_RESET,
        fitness_fn_j       = fitness_fn,
        initial_mu_k       = jnp.array(mu0),
        initial_L_inv_k    = jnp.array(L_inv0),
        context            = context_arr,
        M_inner            = M_INNER,
        block_to_agent_idx = jnp.array(block_to_agent),
        initial_v_k        = jnp.array(v0),
    )

    # ── NumPy 化 ────────────────────────────────────────────────────────
    final_pi_np = np.array(final_pi)
    final_mu_np = np.array(final_mu)
    final_L_np  = np.array(final_L)

    # ── 选择执行分量 ───────────────────────────────────────────────────
    exec_mode = (scenario.exec_mode or EXEC_MODE).lower()
    if exec_mode == "weight_mean":
        exec_k = select_components_by_weight(
            final_pi_np, block_to_agent, prev_exec_k
        )
    elif exec_mode == "cost_select":
        exec_k = select_components_by_cost(
            final_mu_np, context_arr, fitness_fn, block_to_agent, prev_exec_k
        )
    else:
        raise ValueError(f"Unsupported EXEC_MODE: {EXEC_MODE!r}")

    selected_blocks = [
        np.asarray(block, dtype=np.float64)
        for block in decoder.select_blocks_from_components(final_mu_np, exec_k)
    ]
    selected_blocks = np.stack(selected_blocks)

    # ── 跨 MPC 控制序列平滑 ──────────────────────────────────────────────
    # 将本轮选中的序列与上轮实际执行的序列（shift-1）做凸组合，
    # 抑制相邻 MPC 步间的控制跳变。不改 MGIGO 求解器内部。
    if prev_selected_blocks is not None:
        alpha = CTRL_SMOOTH_ALPHA
        selected_blocks = (
            (1 - alpha) * selected_blocks
            + alpha * _shift_sequences(prev_selected_blocks, decoder=decoder)
        )

    decision_sequences = decoder.decode(selected_blocks)

    # ── 执行物理推进 ──────────────────────────────────────────────────
    next_states = advance_one_macro_step(scenario, current_states, decision_sequences)

    # ── 稠密轨迹（JAX → NumPy）─────────────────────────────────────────
    dense_traj_np = dense_rollout_np(scenario, current_states, decision_sequences)

    # ── 可视化采样轨迹（从 current_states 起画，与画面中当前车辆对齐）─
    N_VIZ = 60
    sample_trajs_by_agent = {agent.name: [] for agent in scenario.agents}

    best_prediction = prediction_trajs(scenario, current_states, decision_sequences)
    for name, traj in best_prediction.items():
        sample_trajs_by_agent[name].append(traj)

    for _ in range(N_VIZ - 1):
        z = _sample_block_z(final_pi_np, final_mu_np, final_L_np, rng, solver_spec)
        z_decisions = decoder.decode(z)
        sampled_prediction = prediction_trajs(scenario, current_states, z_decisions)
        for name, traj in sampled_prediction.items():
            sample_trajs_by_agent[name].append(traj)

    # ── 热启动准备 ────────────────────────────────────────────────────
    # mu: warm_start_mu（最优序列 shift-1 → component 0，其余加噪声）
    # L / v: 每步重置 identity/zeros（component 重编号后旧精度矩阵不再有效）
    # v_warm 已删除 — solver 每步从 v_zeros 开始，旧 mixture weight 不传递

    control_sequences_by_block = decoder.block_sequences_by_name(selected_blocks)
    return {
        "control_sequences_by_block": control_sequences_by_block,
        "decision_sequences": decision_sequences,
        "dense_traj":    dense_traj_np,
        "next_states":   next_states,
        "warm":          (final_mu_np, exec_k, selected_blocks),
        "trajectories_by_agent": sample_trajs_by_agent,
        "best_block_ks": exec_k,
        "selected_components": exec_k,
    }
