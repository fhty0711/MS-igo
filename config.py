"""
config.py — shared defaults for the generic MGIGO scenario runner.

Scenario-specific agent/block layouts should live in scenarios/*.py. Keep this
file for physical constants, solver defaults, and reusable visualization knobs.
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  【场景参数】
# ══════════════════════════════════════════════════════════════════════════════
LANE_W  = 3.5    # 单车道宽度（m）
N_LANES = 2      # 车道数
VEH_L   = 5.0    # 车辆长度（m）
VEH_W   = 2.0    # 车辆宽度（m）

LANE_Y          = [0.0, LANE_W]     # 两车道中心 y 坐标
LOWER_LANE_Y    = LANE_Y[0]         # 下车道中心
UPPER_LANE_Y    = LANE_Y[1]         # 上车道中心（换道目标）
TARGET_Y        = UPPER_LANE_Y

ROAD_LENGTH     = 400.0             # 道路总长（m）
SAFE_GAP        = 3.0               # 车身安全间隙（m）


# ══════════════════════════════════════════════════════════════════════════════
#  【时间离散】
# ══════════════════════════════════════════════════════════════════════════════
DT                  = 0.5           # 宏观仿真时间步长（s）
DT_C                = 0.05          # 微观积分步长（s）
SUB_STEPS           = int(round(DT / DT_C))   # 每个宏步内的微观子步数
CONTROL_HORIZON     = 12            # 规划时域步数（H × DT 秒预测窗口）


# ══════════════════════════════════════════════════════════════════════════════
#  【车辆物理参数】
#
#  所有车辆使用统一运动学自行车模型 + 一阶执行器滞后，状态 6 维：
#    s = [x, y, v, ψ, curr_acc, curr_steer]
#    x, y       : 全局位置（m）
#    v          : 速度（m/s）
#    ψ          : 偏航角（yaw，rad）
#    curr_acc   : 当前实际加速度（m/s²），含执行器滞后
#    curr_steer : 当前实际前轮转角（rad），含执行器滞后
#
#  执行器动态（一阶滞后）：
#    ȧ = (a_cmd − a_curr) / TAU_ACC
#    δ̇ = (δ_cmd − δ_curr) / TAU_STEER
#
#  运动学更新（后轮参考点）：
#    β = arctan(LR · tan(δ) / WB)
#    ẋ = v · cos(ψ + β)
#    ẏ = v · sin(ψ + β)
#    ψ̇ = v · cos(β) · tan(δ) / WB
#    v̇ = a
# ══════════════════════════════════════════════════════════════════════════════
WHEEL_BASE      = 2.8           # 轴距（m）
LR              = 1.4           # 质心到后轴距（m）

# 执行器时间常数
TAU_ACC         = 0.25          # 加速度一阶滞后时间常数（s）
TAU_STEER       = 0.20          # 转向一阶滞后时间常数（s）

# 控制量约束
MAX_ACC         = 3.0           # 最大加速度幅值（m/s²）
MAX_STEER       = 0.12          # 最大前轮转角幅值（rad）

# 状态量约束
MAX_SPEED           = 30.0          # 最高速度（m/s）
MAX_CENTRIPETAL_ACC = 2.0           # 最大向心加速度（m/s²，防止侧翻）

# ══════════════════════════════════════════════════════════════════════════════
#  【各车期望速度】
# ══════════════════════════════════════════════════════════════════════════════
V_EGO_DESIRED   = 17.5          # 自车期望巡航速度（m/s）
V_FRONT_DESIRED = 20.0          # 前车期望速度（m/s）
V_REAR_DESIRED  = 17.5          # 后车期望速度（m/s）


# ══════════════════════════════════════════════════════════════════════════════
#  【MGIGO 优化超参数】
# ══════════════════════════════════════════════════════════════════════════════
K_COMP  = 3       # 混合高斯分量数
B_SAMP  = 60      # 每轮采样数
B_ELITE = 25      # 精英样本数
T_OPT   = 300     # MGIGO 迭代次数
ALPHA_T = 0.15    # 自然梯度步长
M_INNER = 30      # 博弈内层 MC 采样数
T_RESET = 300     # 混合权重 reset 周期

# 执行策略：
#   "weight_mean"  : Algorithm 3 风格，选择最高权重 component mean，并加 hysteresis
#   "cost_select"  : 枚举 component mean 后按当前物理 cost 选择
EXEC_MODE = "weight_mean"
HYSTERESIS_BIAS = 0.5
WARM_START_NOISE_STD = 0.15
INIT_MU_NOISE_STD = 0.20
Z_REG_WEIGHT   = 1e-4     # z-space soft barrier：|z| 超过阈值后才罚
Z_REG_THRESH   = 2.0      #   tanh(2.0) ≈ 0.96，超出的 z 对物理命令几乎无区分力
CTRL_SMOOTH_ALPHA = 0.3  # 跨 MPC 控制序列平滑 (0=不平滑, 1=完全锚定上轮计划)


# ══════════════════════════════════════════════════════════════════════════════
#  【Remark 10：分级饱和参数】
#  使用嵌套饱和函数 σ(...σ(f) + Φ₁...) + ΦM 实现约束优先级：
#    σ(x) = x / sqrt(1 + (x / SAT_SCALE)^2)
#  低优先级性能目标先饱和，再叠加更高优先级 penalty。
#  避碰与道路边界同属最高安全层，不再被外层饱和。
# ══════════════════════════════════════════════════════════════════════════════
SAT_SCALE = 80.0   # 饱和尺度：|x| ≪ 80 时近似线性保留梯度，|x| ≫ 80 时趋于饱和


# ══════════════════════════════════════════════════════════════════════════════
#  【默认博弈结构参数】
#  新场景应在 scenarios/*.py 中用 AgentSpec、DecisionSpec、BlockSpec 定义
#  agent 数量、block 数量、block_to_agent 和 block_dims。这里仅保留 highway
#  baseline 对应的默认值，供文档和少量旧配置读取。
# ══════════════════════════════════════════════════════════════════════════════
M_AGENT         = 3
N_BLOCKS        = 4
BLOCK_TO_AGENT  = [0, 0, 1, 2]
BLOCK_DIMS      = (CONTROL_HORIZON,) * N_BLOCKS


# ══════════════════════════════════════════════════════════════════════════════
#  【默认 context 布局】（float32）
#  实际 context 维度由 ScenarioSpec.context_total_dim 推导。
# ══════════════════════════════════════════════════════════════════════════════
N_CARS          = 3
STATE_DIM       = 6            # 每车状态维度
CTX_STATE_DIM   = N_CARS * STATE_DIM   # 18
CTX_REF_DIM     = 3            # 三车速度参考
CTX_TOTAL       = CTX_STATE_DIM + CTX_REF_DIM  # 21


# ══════════════════════════════════════════════════════════════════════════════
#  【碰撞安全参数】（footprint overlap 模型）
#  使用 pairwise_footprint_overlap_cost：
#    将相对位置投影到自车车体坐标系，计算椭圆安全度量
#    (rx / (length + safe_gap))² + (ry / width)² < 1 → 碰撞
# ══════════════════════════════════════════════════════════════════════════════
BND_MARGIN = 0.4    # 路边余量（m）：距路边线的最小安全距离

# 前/后车只看极短前缀会漏掉未来冲突；这里默认检查前 2 秒稠密轨迹。
SAFETY_CHECK_SECONDS = 2.0
SAFETY_CHECK_STEPS = int(round(SAFETY_CHECK_SECONDS / DT_C))


# ══════════════════════════════════════════════════════════════════════════════
#  【仿真参数】
# ══════════════════════════════════════════════════════════════════════════════
N_MPC_STEPS     = 25            # MPC 闭环仿真步数
SNAP_FRAMES     = [0, 12, 24]   # 3 列对比图帧序号
RNG_SEED        = 42            # 随机种子
