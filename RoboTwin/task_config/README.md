# Panthera 任务配置与采集验证

本目录保存 Panthera 任务采集配置，并记录任务级采集验证进度。状态更新截至 2026-08-21。

## 状态口径

- `PASS`：至少有一个测试 seed 完成规划、任务成功判定、数据保存和指令生成。
- “尚未通过”：在当前 smoke 边界内没有成功 episode；可能由规划、任务逻辑、物理稳定性或代码异常造成。
- 单条 smoke 仅用于验证采集链路连通性，不代表所有随机化 seed 或大规模采集的成功率已经验证。
- 当前尚未通过的任务不应直接用于正式批量采集；修复后必须重新完成端到端回归。

## 双臂采集

Panthera 双臂模式使用两台 `panthera-6dof`，动作维度为 14（每臂 6 个关节 + 1 个夹爪）。

截至 2026-08-21，50 个任务中有 41 个至少成功采集过 1 条双臂 smoke。以下 9 个任务尚未通过：

| 任务 | 当前现象 | 当前结论 |
| --- | --- | --- |
| `handover_mic` | 300 秒内 seed 0～111 连续失败，现有日志没有给出具体失败动作 | 尚未通过，需定位交接序列中的规划或成功判定失败 |
| `lift_pot` | 300 秒内 seed 0～191 连续失败，现有日志没有给出具体失败动作 | 尚未通过，需定位双臂同步抓取或抬升阶段 |
| `place_can_basket` | 多个 seed 出现 `target_pose cannot be None for move action` | 尚未通过，需定位未生成有效抓取位姿的具体动作 |
| `place_cans_plasticbox` | 300 秒内 seed 0～17 连续失败，现有日志没有给出具体失败动作 | 尚未通过，需区分规划失败和最终位置判定失败 |
| `place_dual_shoes` | seed 0～181 未成功；同时观察到规划失败和 `041_shoe` 初始物理不稳定 | 尚未通过，规划可达性与物体稳定性都需排查 |
| `put_bottles_dustbin` | 多个 seed 出现 `list index out of range` | 尚未通过，优先修复动作列表下标异常 |
| `open_microwave` | 原始手柄位置和姿态对 Panthera 存在结构性可达性问题 | 尚未通过；不建议通过放宽 Panthera 关节限位解决 |
| `rotate_qrcode` | 精确对齐目标存在双臂 IK 失败；放宽姿态约束后释放物理结果不正确 | 尚未通过，需同时满足规划与最终物理姿态 |
| `stack_bowls_three` | 已观察到右侧碗到中心堆叠目标在失败 seed 上不可达 | 尚未通过，需评估 Panthera 专用场景初始化范围 |

其余 41 个任务都至少通过过 1 条双臂 smoke。其中 `click_alarmclock` 和 `click_bell` 在 Panthera 专用按压路径修复后完成了双臂端到端回归。

## 单臂采集

单臂模式只初始化一台居中的 `panthera-6dof`，动作维度为 7（6 个关节 + 1 个夹爪）。当前代码允许对 `script/collect_data.py` 中明确登记的 32 个单臂语义任务使用 `<task>_panthera_single` 配置名。

历史 smoke 覆盖了其中 31 个任务，27 个任务至少成功采集过 1 条单臂数据。当前仍需处理或重新验证的 5 个任务如下：

| 任务 | 当前状态 |
| --- | --- |
| `adjust_bottle` | 已通过双臂 smoke，但尚未完成单臂 smoke |
| `click_alarmclock` | 历史单臂 smoke 失败；按压逻辑修复后只完成了双臂回归，单臂尚未重新验证 |
| `click_bell` | 历史单臂 smoke 失败；按压逻辑修复后只完成了双臂回归，单臂尚未重新验证 |
| `open_microwave` | 单臂和双臂均未通过，存在结构性可达性问题 |
| `place_phone_stand` | 单臂未通过；放置规划与手机携持、释放的物理稳定性尚未同时满足 |

这里的“尚未重新验证”不同于“已确认无法完成”：`click_alarmclock`、`click_bell` 和 `adjust_bottle` 需要补跑单臂回归后才能更新支持状态。

## 配置命名与生成边界

- 双臂配置：`<task>_panthera.yml`。
- 单臂配置入口：`<task>_panthera_single`。
- 对已登记的单臂任务，采集入口会从对应的 `<task>_panthera.yml` 派生单臂配置，统一设置 `arm_mode: single` 和单个 Panthera embodiment；不需要为每个任务复制一份内容相同的 YAML。
- 只有确实需要覆盖基础参数的任务才应保留显式 `*_panthera_single.yml`，避免重复配置。

## 验证依据

- 2026-08-20 单臂/双臂 smoke：`RoboTwin/data/panthera_single_dual_smoke_20260820/results.tsv`。
- 2026-08-21 对此前未覆盖的 19 个任务补做双臂 smoke：13 个通过，6 个在 300 秒限制内未找到成功 seed。
- `OIDN invalid handle` 是采集过程中观察到的渲染器警告，不作为任务成功或失败的判据。
