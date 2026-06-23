# MGIGO 自动驾驶规划实验框架

这是一个基于 **MGIGO 黑箱优化** 的自动驾驶运动规划实验代码库。项目目标是快速构建、运行和对比不同驾驶场景中的多智能体闭环规划策略，尤其适合测试非光滑、逻辑组合、STL robustness 这类传统梯度优化不方便直接处理的 cost。

当前代码支持：

- 可插拔驾驶场景
- 可插拔 cost profile
- 多智能体 MGIGO / MPC 闭环仿真
- Signal Temporal Logic safety robustness cost
- wrapper / baseline / matched-baseline cost 对比
- 自动生成轨迹图、指标图、视频和验证报告

---

## 1. 项目能做什么

当前主要实验是 **双向两车道借道超车**：

- 自车在右侧车道向前行驶。
- 前方有慢车。
- 左侧为对向车道，有对向车驶来。
- 自车需要判断是否借对向车道完成超车，或者在风险过高时等待。

这个场景用于验证：

- safe case：对向车距离足够远，算法是否能完成借道超车并回道。
- blocked case：对向车太近，算法是否能拒绝危险超车。
- critical case：临界对向间隙下，算法是否能保持安全优先级。

已注册场景：

| 场景名 | 说明 |
|---|---|
| `borrow_overtake_safe` | 安全可超车场景 |
| `borrow_overtake_blocked` | 明显不可超车场景 |
| `borrow_overtake_critical` | 临界可超/不可超场景 |
| `borrow_overtake` | 默认别名，目前指向 critical |
| `highway_merge` | 高速并道场景 |
| `signalized_intersection` | 黄灯路口 dilemma：ego 在 stop/pass 间决策，横向交通为 obey/yellow-rush/red-run 多模态概率模型 |

黄灯路口 benchmark 的详细中文说明见
[`README_signalized_intersection_CN.md`](README_signalized_intersection_CN.md)，其中包含场景定义、prioritized chance/STL cost 设计、ablation 结果和论文层次结论。

---

## 2. 核心特点

### MGIGO 黑箱优化

MGIGO 在控制序列 block 上采样、评价和更新分布，不要求 cost 可微。因此可以直接处理：

- `max/min` robustness
- 分段逻辑
- hard safety gate
- Signal Temporal Logic temporal constraints
- 非光滑 cost transformation

### STL robustness cost

借道超车 cost 中的硬安全约束使用 Signal Temporal Logic robustness 表达。代码中通过 `costs/stl.py` 显式定义原子谓词、`not / and / or / implies`、`G / F / U` 等定量 robustness 语义，并将公式 robustness 转换为 `g(x) <= 0` 形式交给 wrapper 或 matched baseline。

| 约束 | STL 形式 |
|---|---|
| 避碰 | `G[0,T] no_collision` |
| 道路边界 | `G[0,T] inside_road` |
| blocked 中心线约束 | `G[0,T] (blocked_gap -> not_cross_centerline)` |
| 对向车安全 | validated scalar gate for `borrow_lane -> gap/TTC safe` |
| 对向车道低速滞留 | validated scalar gate for `borrow_lane -> v >= v_min` |
| 超车完成 | validated rolling progress gate |
| 回道 | validated return-to-lane gate |

这里没有依赖外部 STL parser，但不是只写“STL 风格”的经验项；当前实现是一个项目内 STL robustness 子集，硬安全公式直接按 STL 定量语义评价整条 rollout。对向车间隙、低速滞留、超车进度和回道仍保留已验证的 exact-penalty 数值尺度，避免一次性改变已通过验证的策略门控行为。MGIGO 不需要这些项可微，因此可以直接优化 `min / max / implication / temporal operator` 与非光滑门控组合后的黑箱 cost。

### Cost Profile 对比

借道超车目前有三套 cost：

| profile | 文件 | 说明 |
|---|---|---|
| `borrow_overtake` | `costs/borrow_overtake.py` | 使用 constraint DSL wrapper |
| `borrow_overtake_baseline` | `costs/borrow_overtake_baseline.py` | 旧手写 hierarchical cost |
| `borrow_overtake_matched` | `costs/borrow_overtake_matched.py` | 不调用 wrapper，但手写复现 wrapper 数学变换 |
| `signalized_intersection` | `costs/signalized_intersection.py` | 单 ego 概率横向交通 + prioritized STL rules |

`matched` 的作用是控制变量：如果 wrapper 和 matched 结果一致，说明效果来自这套 log/saturation/hard/tunable/priority transformation，而不是 API 本身的特殊行为。

---

## 3. 快速运行

推荐在 WSL2 + CUDA GPU 中运行。

```bash
git clone https://github.com/fhty0711/igo.git
cd igo
uv sync
```

运行借道超车 safe case：

```bash
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py borrow_overtake_safe borrow_overtake
```

运行 blocked case：

```bash
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py borrow_overtake_blocked borrow_overtake
```

运行 critical case：

```bash
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py borrow_overtake_critical borrow_overtake
```

运行黄灯路口 dilemma benchmark（推荐 WSL2 Ubuntu + CUDA）：

```bash
wsl.exe -d Ubuntu-22.04 bash -lc 'cd /mnt/d/claude_workspace1/igo && JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python run_mgigo_scenario.py signalized_intersection signalized_intersection'
```

切换 cost profile：

```bash
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py borrow_overtake_critical borrow_overtake_baseline
JAX_PLATFORMS=cuda uv run python run_mgigo_scenario.py borrow_overtake_critical borrow_overtake_matched
```

输出文件会写入：

```text
figures/*.png
figures/*.mp4
```

CPU 也能运行，但速度会明显慢。

---

## 4. Wrapper 验证结果

可以直接查看报告：

```text
reports/wrapper_validation_report/wrapper_validation_report.html
reports/wrapper_validation_report/wrapper_validation_report.docx
reports/wrapper_validation_report.zip
```

HTML 报告包含可播放视频，适合直接发给同事查看。

关键结果：

| 场景 | wrapper | old baseline | matched baseline |
|---|---|---|---|
| safe | 成功且安全 | 成功且安全 | 与 wrapper 一致 |
| blocked | 不超车，安全 | 不超车，安全 | 与 wrapper 一致 |
| critical | 放弃/等待，安全 | 完成超车但对向冲突 | 与 wrapper 一致 |

critical case 中 old baseline 的典型结果：

```text
pass_success = True
conflict_free_while_borrowing = False
min_gap_while_borrowing = -74.8 m
min_TTC_while_borrowing = -2.20 s
```

wrapper / matched 的典型结果：

```text
pass_success = False
conflict_free_while_borrowing = True
min_gap_while_borrowing = 232.6 m
min_TTC_while_borrowing = 9.92 s
```

这说明 wrapper 所表达的 constraint-to-cost transformation 可以更可靠地保持 hard safety priority。matched baseline 与 wrapper 一致，说明效果来自数学变换本身；wrapper 的工程优势是把这套变换结构化、参数化、可复用。

---

## 5. 批量对比实验

运行完整 cost profile 对比：

```bash
JAX_PLATFORMS=cuda uv run python compare_cost_profiles.py --force
```

默认会跑：

```text
borrow_overtake_safe
borrow_overtake_blocked
borrow_overtake_critical
```

并比较：

```text
borrow_overtake
borrow_overtake_baseline
borrow_overtake_matched
```

输出：

```text
figures/cost_profile_comparison/summary.csv
figures/cost_profile_comparison/overview_safety_table.png
figures/cost_profile_comparison/overview_metrics.png
figures/cost_profile_comparison/overview_trajectories.png
```

生成可分享报告：

```bash
python generate_wrapper_report.py
```

输出：

```text
reports/wrapper_validation_report/
reports/wrapper_validation_report.zip
```

---

## 6. 代码结构

```text
run_mgigo_scenario.py          # 单场景运行入口
compare_cost_profiles.py       # wrapper / baseline / matched 批量对比
generate_wrapper_report.py     # 生成 HTML / DOCX / ZIP 报告

scenarios/
  spec.py                      # AgentSpec, DecisionSpec, BlockSpec, ScenarioSpec
  registry.py                  # 场景注册
  borrow_overtake.py           # 借道超车场景族
  highway_merge.py             # 高速并道场景
  signalized_intersection.py   # 黄灯路口 dilemma 场景（单 ego 优化）

costs/
  constraint_dsl.py            # cost wrapper / constraint DSL
  common.py                    # rollout 和通用 cost helper
  registry.py                  # cost profile 注册
  borrow_overtake.py           # wrapper cost
  borrow_overtake_baseline.py  # 旧手写 hierarchical baseline
  borrow_overtake_matched.py   # matched hand-written wrapper transformation
  highway_merge.py             # 高速并道 wrapper cost
  signalized_intersection.py   # 概率横向交通 + prioritized STL rules

backends/
  generic_scenario.py          # 通用闭环仿真 backend

igo/
  MPC_G_MS.py                  # MGIGO 多智能体求解器

planner.py                     # 单步 MPC 规划封装
decision_layout.py             # block <-> named decision 解码
scenario_runtime.py            # 状态推进和预测轨迹
viz_utils.py                   # 道路、车辆、轨迹渲染
config.py                      # 全局求解和车辆参数
```

---

## 7. 调用关系

```text
run_mgigo_scenario.py
  -> scenarios.get_scenario(...)
  -> costs.get_cost_functions(...)
  -> backends.get_backend(...)
  -> GenericScenarioBackend.step()
  -> planner.plan()
  -> BlockDecoder
  -> mmog_igo_rne_blocks_solver()
  -> selected control blocks
  -> scenario_runtime.advance_one_macro_step()
  -> viz_utils.render_agents_panel()
  -> figures/*.png / figures/*.mp4
```

cost 对候选轨迹的计算流程：

```text
joint_sample_flat
  -> BlockDecoder.decode()
  -> dense_rollout_from_decisions()
  -> objective + STL robustness violations
  -> wrapper / baseline / matched transformation
  -> scalar black-box cost
```

---

## 8. 新增场景

新增场景通常需要：

1. 新建 `scenarios/<new_scenario>.py`
2. 定义 `ScenarioSpec`
3. 在 `scenarios/registry.py` 注册
4. 新建或复用 `costs/<new_cost>.py`
5. 在 `costs/registry.py` 注册 cost profile
6. 运行：

```bash
uv run python run_mgigo_scenario.py <scenario_name> <cost_profile>
```

后续可以继续添加：

- 红灯停车场景
- 行人横穿场景
- 黄灯路口 dilemma 的多 agent 非合作扩展
- 多车交互超车场景
- 概率 / CVaR / chance constraint cost
- 更完整的 STL parser / DSL

---

## 9. 依赖

项目使用 `uv` 管理环境。

主要依赖：

- Python
- JAX / CUDA JAX
- NumPy
- Matplotlib
- python-docx
- Pillow

`pyproject.toml` 中记录了当前运行所需的核心依赖。

---

## 10. 参考

本项目参考了以下项目和论文代码：

- https://github.com/qlp71/IOC_AGV
- https://github.com/Konsteidinoeevich/MGIGO
