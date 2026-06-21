"""
models.py — generic vehicle dynamics primitives.

JAX functions are used inside solver/cost rollouts; NumPy functions are used by
closed-loop simulation and visualization.

所有车辆统一使用运动学自行车模型 + 一阶执行器滞后，状态 6 维：
    s = [x, y, v, ψ, curr_acc, curr_steer]
    x, y       : 全局位置（m）
    v          : 速度（m/s）
    ψ          : 偏航角（yaw，rad）
    curr_acc   : 当前实际加速度（m/s²），含执行器滞后
    curr_steer : 当前实际前轮转角（rad），含执行器滞后

执行器动态（一阶滞后）：
    ȧ = (a_cmd − a_curr) / TAU_ACC
    δ̇ = (δ_cmd − δ_curr) / TAU_STEER

运动学更新（后轮参考点）：
    β = arctan(LR · tan(δ) / WHEEL_BASE)
    ẋ = v · cos(ψ + β)
    ẏ = v · sin(ψ + β)
    ψ̇ = v · cos(β) · tan(δ) / WHEEL_BASE   （限制向心加速度）
    v̇ = a
"""

import numpy as np
from functools import partial
from jax import jit, lax
import jax.numpy as jnp

from config import (
    WHEEL_BASE, LR,
    TAU_ACC, TAU_STEER, MAX_ACC, MAX_STEER,
    MAX_SPEED, MAX_CENTRIPETAL_ACC,
)


# ══════════════════════════════════════════════════════════════════════════════
#  JAX 微观单步
# ══════════════════════════════════════════════════════════════════════════════

@jit
def stable_single_car_step(state, target_raw_acc, target_raw_steer, dt):
    """
    ego 运动学自行车模型微观单步（含执行器滞后 + 向心加速度限制）。

    state : (6,)  [x, y, v, ψ, curr_acc, curr_steer]
    target_raw_acc   : 标量，无界加速度指令（tanh 映射前）
    target_raw_steer : 标量，无界转角指令（tanh 映射前）
    dt               : 积分步长（通常 DT_C = 0.05）

    返回 : (6,)  新状态
    """
    x, y, v, psi, curr_acc, curr_steer = (
        state[0], state[1], state[2], state[3], state[4], state[5]
    )

    t_acc = MAX_ACC * jnp.tanh(target_raw_acc)
    t_steer = MAX_STEER * jnp.tanh(target_raw_steer)

    next_acc = curr_acc + (t_acc - curr_acc) / TAU_ACC * dt
    next_acc = jnp.clip(next_acc, -MAX_ACC, MAX_ACC)
    next_steer = curr_steer + (t_steer - curr_steer) / TAU_STEER * dt
    next_steer = jnp.clip(next_steer, -MAX_STEER, MAX_STEER)

    beta = jnp.arctan(LR * jnp.tan(next_steer) / WHEEL_BASE)

    next_v = v + next_acc * dt
    next_v = jnp.clip(next_v, 0.0, MAX_SPEED)

    psi_dot_raw = next_v * jnp.cos(beta) * jnp.tan(next_steer) / WHEEL_BASE
    v_safe = jnp.maximum(next_v, 1e-3)
    max_psi_dot = MAX_CENTRIPETAL_ACC / v_safe
    next_psi_dot = jnp.clip(psi_dot_raw, -max_psi_dot, max_psi_dot)

    next_psi = psi + next_psi_dot * dt
    next_x = x + next_v * jnp.cos(next_psi + beta) * dt
    next_y = y + next_v * jnp.sin(next_psi + beta) * dt

    return jnp.array([next_x, next_y, next_v, next_psi, next_acc, next_steer])


# ══════════════════════════════════════════════════════════════════════════════
#  JAX 控制平滑
# ══════════════════════════════════════════════════════════════════════════════

@jit
def low_pass_filter_sequence(current_sequence, alpha=0.0, init_value=0.0):
    """
    对控制序列做一阶低通滤波（指数滑动平均），抑制相邻 MPC 步间的控制跳变。

    y[t] = α · y[t−1] + (1 − α) · x[t]

    current_sequence : (N,)  原始控制序列
    alpha            : 平滑系数 (0 = 不过滤, →1 = 越平滑)
    init_value       : 初始状态值

    返回 : (N,)  滤波后序列
    """
    def scan_fn(carry, x):
        next_val = alpha * carry + (1.0 - alpha) * x
        return next_val, next_val
    _, filtered_seq = lax.scan(scan_fn, init_value, current_sequence)
    return filtered_seq


# ══════════════════════════════════════════════════════════════════════════════
#  JAX 碰撞代价
# ══════════════════════════════════════════════════════════════════════════════

@jit
def pairwise_footprint_overlap_cost(pos_a, pos_b, psi_a, psi_b,
                                    length=5.0, width=2.0, safe_gap=3.0):
    """
    车体坐标系椭圆碰撞代价。

    将两车相对位置投影到自车车体坐标系，计算椭圆安全度量：
        (rx / (length + safe_gap))² + (ry / width)²

    进入椭圆内（值 < 1）时返回连续违反量惩罚，避免 indicator plateau。

    pos_a, pos_b : (2,)  [x, y]  车辆位置
    psi_a, psi_b : 标量          偏航角（rad）
    length       : 车长（m）
    width        : 车宽（m）
    safe_gap     : 安全间隙（m）

    返回 : 标量  连续碰撞违反代价
    """
    eff_len = length + safe_gap
    eff_wid = width

    def violation_in_body_frame(src_pos, dst_pos, src_psi):
        dx = src_pos[0] - dst_pos[0]
        dy = src_pos[1] - dst_pos[1]

        cos_psi = jnp.cos(src_psi)
        sin_psi = jnp.sin(src_psi)

        rx = dx * cos_psi + dy * sin_psi
        ry = -dx * sin_psi + dy * cos_psi
        overlap_indicator = (rx / eff_len) ** 2 + (ry / eff_wid) ** 2
        return jnp.maximum(1.0 - overlap_indicator, 0.0)

    # Use both vehicle frames so the cost is sensitive to both yaw angles.
    violation_a = violation_in_body_frame(pos_a, pos_b, psi_a)
    violation_b = violation_in_body_frame(pos_b, pos_a, psi_b)
    violation = jnp.maximum(violation_a, violation_b)
    return 10.0 * violation ** 2


# ══════════════════════════════════════════════════════════════════════════════
#  NumPy 版（仿真执行 + 可视化采样）
# ══════════════════════════════════════════════════════════════════════════════

def np_stable_single_car_step(state, target_raw_acc, target_raw_steer, dt):
    """stable_single_car_step 的 NumPy 版。"""
    x, y, v, psi, curr_acc, curr_steer = (
        float(state[0]), float(state[1]), float(state[2]),
        float(state[3]), float(state[4]), float(state[5]),
    )
    t_acc   = MAX_ACC * np.tanh(float(target_raw_acc))
    t_steer = MAX_STEER * np.tanh(float(target_raw_steer))

    next_acc = curr_acc + (t_acc - curr_acc) / TAU_ACC * dt
    next_acc = float(np.clip(next_acc, -MAX_ACC, MAX_ACC))
    next_steer = curr_steer + (t_steer - curr_steer) / TAU_STEER * dt
    next_steer = float(np.clip(next_steer, -MAX_STEER, MAX_STEER))

    beta = np.arctan(LR * np.tan(next_steer) / WHEEL_BASE)

    next_v = v + next_acc * dt
    next_v = float(np.clip(next_v, 0.0, MAX_SPEED))

    psi_dot_raw = next_v * np.cos(beta) * np.tan(next_steer) / WHEEL_BASE
    v_safe = max(abs(next_v), 1e-3)
    max_psi_dot = MAX_CENTRIPETAL_ACC / v_safe
    next_psi_dot = float(np.clip(psi_dot_raw, -max_psi_dot, max_psi_dot))

    next_psi = psi + next_psi_dot * dt
    next_x = x + next_v * np.cos(next_psi + beta) * dt
    next_y = y + next_v * np.sin(next_psi + beta) * dt

    return np.array(
        [next_x, next_y, next_v, next_psi, next_acc, next_steer],
        dtype=np.float64,
    )


