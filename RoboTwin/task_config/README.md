# Panthera 任务配置与采集验证

本目录保存 Panthera 任务采集配置，并记录任务级采集验证进度。状态更新截至 2026-09-03。

## 状态口径

- `PASS`：至少有一个测试 seed 完成规划、任务成功判定、数据保存和指令生成。
- “尚未通过”：在当前 smoke 边界内没有成功 episode；可能由规划、任务逻辑、物理稳定性或代码异常造成。
- 单条 smoke 仅用于验证采集链路连通性，不代表所有随机化 seed 或大规模采集的成功率已经验证。
- “未通过”表示在记录的限时 smoke 窗口内没有找到成功 seed，不等于已经证明任务无法完成。
- 当前尚未通过的任务不应直接用于正式批量采集；修复后必须重新完成端到端回归。

## 双臂采集

Panthera 双臂模式使用两台 `panthera-6dof`，动作维度为 14（每臂 6 个关节 + 1 个夹爪）。

截至 2026-09-04，50 个任务中有 48 个至少成功采集过 1 条双臂 smoke。以下 2 个任务尚未通过：

| 任务 | 当前现象 | 当前结论 |
| --- | --- | --- |
| `place_dual_shoes` | 300 秒内 seed 0～144 均未通过；同时观察到规划失败、`041_shoe` 初始不稳定和一次空目标位姿 | 尚未通过，规划可达性与物体初始化稳定性都需排查 |
| `rotate_qrcode` | 300 秒内 seed 0～154 均未通过，未观察到代码异常 | 尚未通过，需定位精确对齐动作的双臂规划边界，同时保持最终物理姿态正确 |

`handover_mic` 已完成 Panthera 双臂正式采集验证：在不改变场景随机化的前提下，33 个随机 seed 中筛得 20 个有效 episode，并完成 20 个 HDF5、60 个视频和 20 条场景信息的文件级检查。当前 seed 筛选通过率约为 60.6%；该任务可用于阶段性批量采集，后续如调整交接策略仍需重新回归。

`open_microwave` 已完成单双臂 Panthera 闭环验证：双臂配置和单臂配置分别抽测 seed 0～9，均为 10/10 通过。修复包括绕过 Panthera 的零距离抓取覆盖、使用可达的微波炉接触点，以及沿门铰链生成累计拉动目标；物体模型随机选择和场景随机化保持不变。

其余 48 个任务都至少通过过 1 条双臂 smoke。其中 `click_alarmclock` 和 `click_bell` 在 Panthera 专用按压路径修复后完成了双臂端到端回归；此前未通过的 `place_can_basket`、`place_cans_plasticbox` 和 `put_bottles_dustbin` 已分别以 seed 17、11、17 完成端到端采集。

`lift_pot` 已通过 Panthera 双臂端到端回归。Panthera 模式仅启用锅模型 0，并将双臂抓取位置沿手指方向深入 10 mm；锅的 XY 位置与旋转随机化仍保留。锅模型 1 的锅柄更低，当前侧向预抓取会使 Panthera 左臂第 4 关节到达约 1.6000 rad 上限，直接进入最终位姿还会推撞锅体，因此暂不用于 Panthera 采集。该限制不影响其他 embodiment 的原始模型随机逻辑。

此前重点验证的双臂协调任务中，`grab_roller`、`handover_block`、`hanging_mug`、`handover_mic`、`lift_pot`、`pick_dual_bottles`、`place_cans_plasticbox`、`put_bottles_dustbin` 和 `stack_bowls_three` 已通过；`place_dual_shoes` 尚未通过。仓库没有独立的“协调任务”标签，这里的分类依据任务语义和左右臂动作序列。

## 单臂采集

单臂模式只初始化一台居中的 `panthera-6dof`，动作维度为 7（6 个关节 + 1 个夹爪）。当前代码允许对 `script/collect_data.py` 中明确登记的 32 个单臂语义任务使用 `<task>_panthera_single` 配置名。

截至 2026-09-04，32 个单臂任务均已执行过 smoke，32 个均至少成功采集过 1 条单臂数据。当前没有单臂尚未通过任务：

| 任务 | 当前现象 | 当前结论 |
| --- | --- | --- |

`adjust_bottle` 已完成 5 条单臂完整采集与文件级检查；`click_alarmclock`、`click_bell`、`place_phone_stand` 和 `open_microwave` 的单臂回归也已通过。`rotate_qrcode` 当前为单臂通过、双臂尚未通过，说明问题集中在双臂配置下的规划或场景边界，而不是任务链路整体失效。

## 完整配置状态索引

双臂 `PASS`（48）：

```text
adjust_bottle, beat_block_hammer, blocks_ranking_rgb, blocks_ranking_size,
click_alarmclock, click_bell, dump_bin_bigbin, grab_roller, handover_block, handover_mic, open_microwave,
hanging_mug, lift_pot, move_can_pot, move_pillbottle_pad, move_playingcard_away,
move_stapler_pad, open_laptop, pick_diverse_bottles, pick_dual_bottles,
place_a2b_left, place_a2b_right, place_bread_basket, place_bread_skillet,
place_burger_fries, place_can_basket, place_cans_plasticbox,
place_container_plate, place_empty_cup, place_fan, place_mouse_pad,
place_object_basket, place_object_scale, place_object_stand, place_phone_stand,
place_shoe, press_stapler, put_bottles_dustbin, put_object_cabinet, scan_object,
shake_bottle, shake_bottle_horizontally, stack_blocks_three, stack_blocks_two,
stack_bowls_three, stack_bowls_two, stamp_seal, turn_switch
```

双臂尚未通过（2）：

```text
place_dual_shoes, rotate_qrcode
```

单臂 `PASS`（32）：

```text
adjust_bottle, beat_block_hammer, blocks_ranking_rgb, blocks_ranking_size, open_microwave,
click_alarmclock, click_bell, move_can_pot, move_pillbottle_pad,
move_playingcard_away, move_stapler_pad, open_laptop, place_a2b_left,
place_a2b_right, place_container_plate, place_empty_cup, place_fan,
place_mouse_pad, place_object_scale, place_object_stand, place_phone_stand,
place_shoe, press_stapler, rotate_qrcode, shake_bottle,
shake_bottle_horizontally, stack_blocks_three, stack_blocks_two,
stack_bowls_three, stack_bowls_two, stamp_seal, turn_switch
```

单臂尚未通过（0）：

```text
（无）
```

## 配置命名与生成边界

- 双臂配置：`<task>_panthera.yml`。
- 单臂配置入口：`<task>_panthera_single`。
- 对已登记的单臂任务，采集入口会从对应的 `<task>_panthera.yml` 派生单臂配置，统一设置 `arm_mode: single` 和单个 Panthera embodiment；不需要为每个任务复制一份内容相同的 YAML。
- 只有确实需要覆盖基础参数的任务才应保留显式 `*_panthera_single.yml`，避免重复配置。
- `adjust_bottle_panthera_single.yml` 是保留的显式单臂采集配置，记录该任务已经实测的单臂 smoke 参数。
