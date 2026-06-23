# MGIGO 自动驾驶规划实验框架

这是一个基于 **MGIGO 黑箱优化** 的自动驾驶运动规划实验代码库。项目目标是快速构建、运行和对比不同驾驶场景中的多智能体闭环规划策略，尤其适合测试非光滑、逻辑组合、STL 这类传统梯度优化不方便直接处理的 cost。

当前代码支持：

- 可插拔驾驶场景
- 可插拔 cost profile
- 多智能体 MGIGO / MPC 闭环仿真
- STL-style safety robustness cost
- wrapper / baseline / matched-baseline cost 对比
- 自动生成轨迹图、指标图、视频和验证报告

---

## 1. 项目能做什么

当前有两个场景
1. 高速并道场景

2. 双向两车道借道超车：
  - 自车在右侧车道向前行驶。
  - 前方有慢车。
  - 左侧为对向车道，有对向车驶来。
  - 自车需要判断是否借对向车道完成超车，或者在风险过高时等待。

借道超车场景用于验证：

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

---

## 2. 核心特点

### MGIGO 黑箱优化

MGIGO 不要求 cost 可微。因此可以直接处理：

- `max/min` robustness
- 分段逻辑
- hard safety gate
- STL-style temporal constraints
- 非光滑 cost transformation

借道超车 cost 中包含多类 STL-style 时序约束：G[0,T] no_collision等


### Cost Profile 对比

借道超车目前有三套 cost：

| profile | 文件 | 说明 |
|---|---|---|
| `borrow_overtake` | `costs/borrow_overtake.py` | 使用 constraint DSL wrapper |
| `borrow_overtake_baseline` | `costs/borrow_overtake_baseline.py` | 旧手写 hierarchical cost |
| `borrow_overtake_matched` | `costs/borrow_overtake_matched.py` | 不调用 wrapper，但手写复现 wrapper 数学变换 |

`matched` 的作用是控制变量：如果 wrapper 和 matched 结果一致，说明效果来自这套 log/saturation/hard/tunable/priority transformation，而不是 API 本身的特殊行为。

---

## 3. 快速运行

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

这说明 wrapper 所表达的 constraint-to-cost transformation 可以更可靠地保持 hard safety priority。matched baseline 与 wrapper 一致，wrapper 的工程优势是把这套变换结构化、参数化、可复用。

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

生成报告：

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

costs/
  constraint_dsl.py            # cost wrapper / constraint DSL
  common.py                    # rollout 和通用 cost helper
  registry.py                  # cost profile 注册
  borrow_overtake.py           # wrapper cost
  borrow_overtake_baseline.py  # 旧手写 hierarchical baseline
  borrow_overtake_matched.py   # matched hand-written wrapper transformation
  highway_merge.py             # 高速并道 wrapper cost

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



## 7. 新增场景

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
- 多车交互超车场景
- 概率 / CVaR / chance constraint cost
- 更规范的 STL DSL

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

## 10. 参考
本项目参考了https://github.com/qlp71/IOC_AGV  和   https://github.com/Konsteidinoeevich/MGIGO

