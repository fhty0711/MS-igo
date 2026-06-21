# MGIGO 自动驾驶场景规划代码说明

这个仓库是从原始 `IOC_AGV` 项目中单独抽出来的 **MGIGO 场景规划核心代码**。  
目的不是保留整个大项目，而是方便在另一台电脑上阅读、运行和继续开发自动驾驶规划实验。

本仓库只保留当前 active 链路：

- MGIGO 多智能体黑箱优化器
- 可插拔场景 `ScenarioSpec`
- 可插拔 cost profile
- constraint-DSL 代价函数
- 闭环 MPC 仿真
- 通用可视化输出

旧 IOC、MRS、汇报文件、旧 highway 脚本、旧经验 cost、视频输出都没有放进这个仓库。

---

## 1. 当前能跑什么

目前已经注册并验证的场景是：

| 名称 | 文件 | 说明 |
|---|---|---|
| `highway_merge` | `scenarios/highway_merge.py` | 自车从下车道并入上车道，目标车道有前车和后车 |

对应 cost profile：

| 名称 | 文件 | 说明 |
|---|---|---|
| `highway_merge` | `costs/highway_merge.py` | 使用 constraint-DSL 组织 objective 和安全/终端约束 |

运行后输出：

```text
figures/mgigo_highway_snapshot.png
figures/mgigo_highway.mp4
```

---

## 2. 推荐运行方式

推荐在 WSL2 + CUDA GPU 中运行：

```bash
git clone https://github.com/fhty0711/igo.git
cd igo
uv sync
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py highway_merge highway_merge
```

如果不显式写场景和 cost，也会默认运行当前 highway 场景：

```bash
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py
```

CPU 也能运行，但会明显更慢。之前在 WSL2 + RTX 3070 Laptop GPU 上测到：

```text
25 步规划循环：约 60 秒
完整运行，包括 snapshot 和 mp4：约 88 秒
```

首次运行如果 `uv` 需要创建 `.venv` 或安装依赖，会额外花几分钟，这不属于算法本身耗时。

---

## 3. 总体调用链

当前主调用链如下：

```text
run_mgigo_scenario.py
  |
  |-- scenarios/registry.py
  |     -> get_scenario("highway_merge")
  |     -> scenarios/highway_merge.py::make_scenario()
  |     -> 返回 ScenarioSpec
  |
  |-- costs/registry.py
  |     -> get_cost_functions("highway_merge")
  |     -> costs/highway_merge.py
  |     -> 返回 ego/front/rear 三个 agent cost
  |
  |-- backends/registry.py
  |     -> get_backend("generic_scenario")
  |     -> GenericScenarioBackend
  |
  v
GenericScenarioBackend.step()
  |
  v
planner.py::plan()
  |
  |-- BlockDecoder(scenario)
  |     -> 把 MGIGO block 解码成 ego_acc、ego_steer 等命名控制序列
  |
  |-- make_fitness_fn(cost_profile)
  |     -> 构造 agent_idx 到对应 cost 函数的分发器
  |
  |-- igo/MPC_G_MS.py::mmog_igo_rne_blocks_solver()
  |     -> 多智能体 MGIGO 求解器
  |     -> 采样 block
  |     -> 调用 fitness_fn(agent_idx, joint_sample, context)
  |
  |-- costs/highway_merge.py
  |     -> 解码候选控制
  |     -> dense_rollout_from_decisions()
  |     -> constraint_dsl.build() 组装后的 cost
  |
  |-- component_selection.py
  |     -> 选择最终执行的 GMM component
  |
  |-- scenario_runtime.py
  |     -> advance_one_macro_step()
  |     -> prediction_trajs()
  |
  v
GenericScenarioBackend 记录历史轨迹
  |
  v
viz_utils.py::render_agents_panel()
  |
  v
figures/*.png / figures/*.mp4
```

---

## 4. 目录和文件作用

### 入口

```text
run_mgigo_scenario.py
```

项目运行入口。负责：

- 读取命令行参数
- 根据名字获取 scenario 和 cost profile
- 创建 backend
- 执行闭环 MPC 仿真
- 保存三列 snapshot
- 保存 mp4 或 gif

常用命令：

```bash
uv run python run_mgigo_scenario.py
uv run python run_mgigo_scenario.py highway_merge highway_merge
```

---

### 场景定义

```text
scenarios/spec.py
scenarios/highway_merge.py
scenarios/registry.py
```

`scenarios/spec.py` 定义通用数据结构：

| 类 | 作用 |
|---|---|
| `AgentSpec` | 描述一个车辆/智能体，包括名字、角色、动力学类型、状态索引、速度参考索引 |
| `DecisionSpec` | 描述一个优化变量，例如 `ego_acc`、`ego_steer` |
| `BlockSpec` | 描述一个 MGIGO block，以及这个 block 内包含哪些 decision |
| `RoadSpec` | 道路和车道参数 |
| `VehicleGeometrySpec` | 车长、车宽、安全间隙 |
| `ScenarioSpec` | 一个完整场景，包括车辆、初始状态、速度参考、block 布局、道路参数等 |

`scenarios/highway_merge.py` 是当前高速并道场景：

- 自车 `ego`：下车道起步，目标是并入上车道
- 前车 `front`：目标车道前方车辆
- 后车 `rear`：目标车道后方车辆

当前 block 布局：

| block | owner agent | decision |
|---|---|---|
| 0 | ego | `ego_acc` |
| 1 | ego | `ego_steer` |
| 2 | front | `front_acc` |
| 3 | rear | `rear_acc` |

`scenarios/registry.py` 负责注册场景名字：

```python
SCENARIO_FACTORIES = {
    "highway_merge": make_highway_merge,
}
```

---

### block 和 decision 解码

```text
decision_layout.py
```

MGIGO 求解器只认识匿名 block 数组，例如：

```text
(n_blocks, solver_width)
```

但是 cost 和 runtime 更适合使用命名控制序列，例如：

```python
{
    "ego_acc": ...,
    "ego_steer": ...,
    "front_acc": ...,
    "rear_acc": ...,
}
```

`BlockDecoder` 就是这两个表示之间的桥：

- `decode()`：block 数组 -> 命名 decision
- `encode()`：命名 decision -> block 数组
- `shift_blocks()`：MPC warm start 时把控制序列前移一拍
- `select_blocks_from_components()`：从 GMM component 均值中选出实际执行的 block

---

### planner

```text
planner.py
component_selection.py
```

`planner.py` 是单步 MPC 规划的核心包装层。它不直接写具体场景 cost，而是通过 scenario 和 cost profile 组织求解。

主要流程：

1. 根据 scenario 创建 `BlockDecoder`
2. 打包 `context_arr`
3. 创建 `fitness_fn`
4. 调用 `mmog_igo_rne_blocks_solver()`
5. 选择最终执行的 component
6. 解码控制序列
7. 推进真实闭环状态
8. 生成可视化预测轨迹
9. 返回 warm start 信息

`component_selection.py` 负责从 MGIGO 的多个 GMM component 中选择要执行的 component。

当前支持：

- `weight_mean`：按 GMM 权重选择
- `cost_select`：枚举 component mean 后按 cost 选择

具体模式由 `config.py` 中的 `EXEC_MODE` 控制。

---

### MGIGO 求解器

```text
igo/MPC_G_MS.py
```

这是多智能体 blockwise MGIGO 求解器。核心逻辑包括：

- 每个 block 用高斯混合分布表示搜索分布
- 按 block 采样候选控制序列
- 按 agent 计算期望 cost
- 根据精英样本更新 GMM 参数
- 输出最终的 component 均值、精度矩阵和权重

planner 调用的是：

```python
mmog_igo_rne_blocks_solver(...)
```

---

### cost 设计

```text
costs/constraint_dsl.py
costs/common.py
costs/highway_merge.py
costs/registry.py
```

`costs/constraint_dsl.py` 提供 constraint-to-cost 工具。基本思想是：

```text
objective(x, ctx)
constraints: g(x, ctx) <= 0
```

然后用 `build()` 把 objective 和约束层组装成 MGIGO 可直接评价的黑箱标量 cost。

支持的约束类型包括：

- `Deterministic`
- `Chance`
- `Robust`
- `DRO`

当前 highway cost 在 `costs/highway_merge.py` 中，注册了三个 agent cost：

```python
ego_cost(...)
front_cost(...)
rear_cost(...)
```

其中 ego cost 包含：

- 速度跟踪
- 目标车道跟踪
- 横向速度惩罚
- 航向角惩罚
- 控制幅值和平滑
- 终端并道质量
- 碰撞违反
- 道路边界违反

front / rear cost 主要包含：

- 速度跟踪
- 加速度平滑
- 与 ego 的短时碰撞检测
- rear 额外包含与 front 的 headway 约束

`costs/registry.py` 注册 cost profile：

```python
COST_PROFILES = {
    "highway_merge": (
        highway_merge.ego_cost,
        highway_merge.front_cost,
        highway_merge.rear_cost,
    ),
}
```

---

### 物理推进和可视化

```text
scenario_runtime.py
models.py
viz_utils.py
backends/generic_scenario.py
```

`models.py` 定义车辆动力学：

- `stable_single_car_step()`：JAX 版自行车模型
- `np_stable_single_car_step()`：NumPy 版自行车模型
- `pairwise_footprint_overlap_cost()`：车体 footprint 碰撞代价

`scenario_runtime.py` 负责把命名控制序列实际 rollout：

- `advance_one_macro_step()`：只执行 MPC 序列第一个控制，推进真实闭环状态
- `prediction_trajs()`：生成整条预测轨迹
- `dense_rollout_np()`：返回 `(time, agent, state_dim)` 格式轨迹

`backends/generic_scenario.py` 负责闭环运行状态：

- 当前状态
- warm start
- 状态历史
- 预测轨迹历史
- 渲染调用

`viz_utils.py` 负责画图：

- 道路
- 车辆矩形
- 历史轨迹
- MGIGO 采样轨迹云
- 最优预测轨迹

---

## 5. 新建一个场景需要改什么

假设要新增借道超车场景 `borrow_overtake`，通常分 4 步。

### 第一步：新建场景文件

新建：

```text
scenarios/borrow_overtake.py
```

里面写：

```python
def make_scenario() -> ScenarioSpec:
    return ScenarioSpec(...)
```

重点填写：

- `initial_states`
- `v_refs`
- `target_y`
- `agents`
- `decisions`
- `blocks`
- `road`
- `vehicle_geometry`
- `snap_labels`
- `cost_profile`

### 第二步：注册场景

修改：

```text
scenarios/registry.py
```

加入：

```python
from .borrow_overtake import make_scenario as make_borrow_overtake

SCENARIO_FACTORIES = {
    "highway_merge": make_highway_merge,
    "borrow_overtake": make_borrow_overtake,
}
```

### 第三步：新建 cost

新建：

```text
costs/borrow_overtake.py
```

至少提供和 agent 数量一致的 cost 函数，例如：

```python
ego_cost(...)
slow_lead_cost(...)
oncoming_cost(...)
```

建议继续用：

```python
from .constraint_dsl import Deterministic, Chance, Robust, DRO, build
```

### 第四步：注册 cost

修改：

```text
costs/registry.py
```

加入：

```python
from . import highway_merge, borrow_overtake

COST_PROFILES = {
    "highway_merge": (...),
    "borrow_overtake": (
        borrow_overtake.ego_cost,
        borrow_overtake.slow_lead_cost,
        borrow_overtake.oncoming_cost,
    ),
}
```

然后运行：

```bash
uv run python run_mgigo_scenario.py borrow_overtake borrow_overtake
```

普通道路场景通常不需要改：

- `planner.py`
- `igo/MPC_G_MS.py`
- `backends/generic_scenario.py`

只有当新场景需要新的动力学、状态维度或特殊画图时，才需要扩展：

- `models.py`
- `scenario_runtime.py`
- `viz_utils.py`

---

## 6. 代码阅读建议

如果只是想理解代码，建议按这个顺序读：

1. `README.md`
2. `ARCHITECTURE.md`
3. `scenarios/spec.py`
4. `scenarios/highway_merge.py`
5. `decision_layout.py`
6. `run_mgigo_scenario.py`
7. `backends/generic_scenario.py`
8. `planner.py`
9. `costs/highway_merge.py`
10. `costs/constraint_dsl.py`
11. `scenario_runtime.py`
12. `igo/MPC_G_MS.py`

这样读会比较顺：先理解场景接口，再理解 block 解码，再看 planner 如何调用 MGIGO，最后看 cost 和求解器细节。

---

## 7. 注意事项

- `costs/highway_merge.py` 现在是新的 constraint-DSL cost，不是旧经验 cost。
- 旧经验 cost 没有放进这个仓库。
- `figures/` 被 `.gitignore` 忽略，运行后生成的视频不会自动提交。
- 如果另一台电脑没有 CUDA，可以去掉 `JAX_PLATFORMS=cuda`，但运行会慢很多。
- 如果 `uv sync` 在 Windows 原生环境遇到 JAX/CUDA 问题，建议改用 WSL2 Ubuntu。

