# Signalized Intersection Benchmark 中文说明

本文档说明当前仓库中新加入的主 benchmark：

```text
Signalized intersection dilemma with probabilistic cross-traffic
and prioritized STL rules
```

它的目标不是再做一个普通路口仿真，而是构造一个能突出 MG-IGO 优势的验证场景：规划器面对黄灯路口时，需要在 `stop` 和 `pass` 两种行为模式之间选择；横向交通不是确定性的单车轨迹，而是由 `obey / yellow-rush / red-run` 组成的多模态概率行为模型；cost 保留黑箱、非光滑、多模态、概率约束和优先级结构，不把问题改写成可微、凸、LQ、MILP 或单峰 surrogate。

---

## 1. 已实现内容

当前实现的是 Scheme A：

- ego 是唯一优化 agent。
- 横向车作为外生概率行为模型参与风险评估和可视化。
- 横向车暂时不作为 active optimizing agent，不进入多智能体 RNE 求解。
- 场景、cost、运行、报告和可视化都走现有 `igo/` 主架构。

相关入口如下：

| 功能 | 文件 |
|---|---|
| 场景定义 | `scenarios/signalized_intersection.py` |
| 场景注册 | `scenarios/registry.py` |
| 主 cost profile | `costs/signalized_intersection.py` |
| ablation cost profiles | `costs/signalized_intersection_no_chance.py`, `costs/signalized_intersection_single_mode.py`, `costs/signalized_intersection_soft_dilemma.py` |
| cost 注册 | `costs/registry.py` |
| 单场景运行 | `run_mgigo_scenario.py` |
| 批量对比 | `compare_signalized_intersection_profiles.py` |
| 报告生成 | `generate_signalized_intersection_report.py` |
| 路口可视化 | `viz_signalized.py`, `viz_utils.py` |
| 回归测试 | `tests/test_signalized_intersection_helpers.py`, `tests/test_constraint_dsl_semantics.py` |

---

## 2. 场景是什么

场景是一个信号灯控制的十字路口：

- ego 沿东西向道路接近路口。
- 信号灯从绿灯进入黄灯，再进入红灯。
- ego 必须决定：
  - 在停止线前停车；
  - 或者在红灯前合法通过并清空路口。
- 南北向横向交通存在概率行为：
  - `obey`：遵守信号，停在冲突区外；
  - `yellow-rush`：黄灯抢行；
  - `red-run`：闯红灯或晚进入冲突区。
- 横向交通轨迹用于 chance risk 和 visualization，但不作为可优化 agent。

已注册三个场景变体：

| 场景 | 含义 | 期望行为 |
|---|---|---|
| `signalized_intersection_easy_pass` | 黄灯较长、ego 速度较高 | 合法安全通过 |
| `signalized_intersection_must_stop` | 黄灯较短、ego 更应停车 | 停止线前停车 |
| `signalized_intersection_critical` | nominal dilemma timing | 在 stop/pass 中选择一个安全合法解 |

道路几何已经重建为横向和纵向双向多车道形式。ego 位于自己的车道中心，横向车概率轨迹云也按车道几何绘制，不再是早期简化的单线示意。

---

## 3. 为什么这个场景适合验证 MG-IGO

这个 benchmark 有几个传统优化不容易自然处理的结构：

1. `stop/pass` 是多模态决策，不应提前指定模式。
2. 红灯合法性、路口阻塞、碰撞风险都带有 temporal min/max 和逻辑 gate。
3. 横向交通是多模态概率分布，不是单一高斯轨迹。
4. chance constraint 需要对每个行为样本计算真实 `g(x, xi, ctx) <= 0` violation。
5. safety 和 legality 必须高优先级，不应被 comfort/progress objective 随意交换。

MG-IGO 以黑箱 cost 排序采样轨迹，不要求 cost 可微，因此可以直接保留这些非光滑结构。

---

## 4. Cost 设计

主 cost profile 是：

```text
costs/signalized_intersection.py
```

它使用本仓库的 Constraint DSL：

```python
objective(x, ctx) + constraints g(x, ctx) <= 0
```

所有 constraint 都保持 `g <= 0` 为满足，`g > 0` 为违反。最终由 `costs/constraint_dsl.py` 按 Constran / ObjectiveComposer 风格进行优先级嵌套、log compression 和 odd saturation，得到 MG-IGO 可以直接排序的 scalar black-box cost。

### 4.1 Hard priority 层

最高优先级是 hard safety / legality：

| 约束 | 类型 | priority | 作用 |
|---|---|---:|---|
| `red_light` | Deterministic hard | 1 | 红灯后不允许处于停止线与路口出口之间；若红灯在预测窗内到来，要求已经停车或已经清空 |
| `road_boundary` | Deterministic hard | 1 | ego 车体必须保持在道路范围内 |
| `no_blocking_intersection` | Deterministic hard | 1 | 不能在路口冲突区内低速停滞，避免 blocking intersection |

这三项是外层高优先级，违反时应压过 lower-priority objective 和 preference。

### 4.2 Probabilistic cross-traffic chance 层

横向交通风险是：

| 约束 | 类型 | priority | 作用 |
|---|---|---:|---|
| `cross_traffic_chance` | Chance tunable | 2 | 对 obey / yellow-rush / red-run 样本逐个计算 AABB penetration violation，并取 `alpha=0.1` 的上分位风险 |

关键点：

- `g_fn(x, xi, ctx)` 保持 per-sample violation。
- 每个 `xi` 都对应一条横向车行为轨迹。
- 不是先把横向交通压成单峰平均轨迹。
- optimization 使用 40 个 deterministic stratified behavior samples。
- evaluation 使用 80 个样本做可视化与指标评估。

### 4.3 Stop/pass dilemma task 层

任务约束是：

| 约束 | 类型 | priority | 作用 |
|---|---|---:|---|
| `dilemma_task` | Deterministic tunable | 3 | 要么在停止线前停车，要么清空路口 |

这里 deliberately 使用 tunable，而不是 weak soft。原因是 stop/pass 是 benchmark 的核心任务，不应只作为很弱的舒适偏好；但它又低于 hard legality / safety 和 chance risk，避免为了完成任务而牺牲安全。

### 4.4 Objective 只做 mild basin

objective 不承担完整 stop/pass 语义，只提供较轻的 basin：

- stop basin：末端靠近停止线前并低速；
- pass basin：末端越过路口出口；
- lane center；
- heading；
- speed reference；
- control effort / smoothness。

真正的任务和安全语义由 prioritized constraints 表达，而不是塞进一个单一 LQ-like objective。

---

## 5. Ablation profiles

为了证明 full design 的必要性，当前包含三个 ablation：

| profile | 改动 | 用来验证什么 |
|---|---|---|
| `signalized_intersection` | full prioritized chance/STL profile | 主方案 |
| `signalized_intersection_no_chance` | 去掉 cross-traffic chance layer | 概率横向交通风险是否必要 |
| `signalized_intersection_single_mode` | 将多模态横向交通替换成单一 deterministic sample | 多模态建模是否必要 |
| `signalized_intersection_soft_dilemma` | 将 stop/pass dilemma 从 tunable 降为 soft | dilemma task priority 是否必要 |

这些 ablation 不是为了调参，而是为了拆解 cost 结构本身的贡献。

---

## 6. 可视化

当前可视化朝 nuPlan 风格迭代，重点是清楚表达场景语义，而不是只画一条轨迹：

- 双向多车道路口；
- ego 车体、历史轨迹和预测轨迹；
- 停止线；
- crosswalk；
- conflict box；
- 当前信号灯相位；
- 横向车概率轨迹云；
- 关键风险样本；
- mode / clearance / red legality / no-blocking 等指标 overlay。

报告中的主要图在：

```text
reports/signalized_intersection_report/assets/overview_trajectories.png
reports/signalized_intersection_report/assets/overview_metrics.png
reports/signalized_intersection_report/assets/overview_outcomes.png
```

---

## 7. 实验结果

完整矩阵包括 3 个场景和 4 个 cost profile，共 12 次闭环运行。结果见：

```text
reports/signalized_intersection_report/signalized_intersection_report.md
reports/signalized_intersection_report/signalized_intersection_report.html
reports/signalized_intersection_report/assets/summary.csv
reports/signalized_intersection_report/assets/manifest.json
```

本次结果的 manifest 显示运行环境为：

```text
scheme: A
scope: single optimizing ego with exogenous probabilistic cross traffic
jax_devices: cuda:0
dt: 0.5
rng_seed: 42
optimization_cross_traffic_samples: 40
evaluation_cross_traffic_samples: 80
```

### 7.1 Full profile 结果

| 场景 | 行为 | task success | safety success | paper claim |
|---|---|---:|---:|---|
| easy_pass | pass | True | True | safe_pass |
| must_stop | stop | True | True | safe_stop |
| critical | pass | True | True | safe_pass |

full profile 的 aggregate：

| cost | task rate | safety rate | scheme A rate | min clearance | failure reason |
|---|---:|---:|---:|---:|---|
| full | 1.00 | 1.00 | 1.00 | 0.21 m | none |

这说明主方案在三个代表性场景中都完成了 stop/pass 任务，并满足红灯合法、no-blocking 和 cross-traffic clearance。

### 7.2 Ablation 结果

| profile | task rate | safety rate | scheme A rate | 主要失败原因 |
|---|---:|---:|---:|---|
| no_chance | 0.67 | 0.00 | 0.00 | cross_traffic_conflict, red_illegal |
| single_mode | 0.67 | 0.67 | 0.67 | red_illegal |
| soft_dilemma | 1.00 | 0.67 | 0.67 | cross_traffic_conflict |

这些结果说明：

- 去掉 chance layer 后，规划会低估横向交通风险，在 easy/critical 中出现 cross-traffic conflict，并在 must_stop 中出现 red illegal。
- 用 single mode 替代多模态交通后，critical 场景会错估横向交通和信号时序，产生 red illegal。
- 将 dilemma task 降为 soft 后，critical 场景虽然完成 pass，但 clearance 变成负值，说明任务偏好过弱会破坏安全余量。

---

## 8. 能证明什么

当前 Scheme A 可以支持以下结论：

1. MG-IGO 可以直接优化 black-box prioritized chance/STL cost。
2. cost 中保留的 `min/max` temporal rule、AABB penetration、red-light legality、no-blocking 和 chance quantile 不需要被改写成可微或凸 surrogate。
3. full prioritized design 在 easy pass、must stop 和 critical dilemma 三类场景中都能得到安全合法行为。
4. ablation 表明：
   - 概率横向交通 chance layer 是必要的；
   - 多模态行为建模比 single-mode surrogate 更稳健；
   - dilemma task 不应降成过弱 soft preference；
   - hard/tunable/soft priority 层次对结果有实质影响。

需要注意：当前结果证明的是 Scheme A，即 single optimizing ego + exogenous probabilistic cross traffic。它还不声称已经实现 active multi-agent non-cooperative RNE intersection game。横向车作为可优化 agent 的方案属于下一阶段。

---

## 9. 如何复现实验

推荐在 WSL2 Ubuntu + CUDA 环境运行。当前机器上的验证命令为：

```bash
cd /mnt/d/claude_workspace1/igo
JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python compare_signalized_intersection_profiles.py --force
python generate_signalized_intersection_report.py
```

如果只想快速刷新已有缓存对应的 summary、figures 和 manifest：

```bash
cd /mnt/d/claude_workspace1/igo
JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python compare_signalized_intersection_profiles.py
python generate_signalized_intersection_report.py
```

单场景运行：

```bash
JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python run_mgigo_scenario.py signalized_intersection_critical signalized_intersection
```

轻量验证：

```bash
python tests/test_constraint_dsl_semantics.py
python tests/test_signalized_intersection_helpers.py
python -m py_compile costs/signalized_intersection.py compare_signalized_intersection_profiles.py generate_signalized_intersection_report.py planner.py scenarios/spec.py scenarios/signalized_intersection.py viz_signalized.py viz_utils.py backends/generic_scenario.py
```

---

## 10. 下一步

建议后续按以下顺序继续：

1. 增强报告中的 paper figure 排版，例如生成更适合论文插图的单页 PDF。
2. 扩展横向交通行为分布，加入更丰富的 arrival-time / speed uncertainty。
3. 增加 repeated seeds / repeated rollouts，统计 success rate 的置信区间。
4. 加入 Scheme B：横向车进入 `ScenarioSpec` 作为 frozen physical participants。
5. 最后再进入 Scheme C：横向车作为 active optimizing agents，扩展成真正多智能体非合作路口博弈。
