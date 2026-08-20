# Panthera Policy 数据转换与训练

本目录承载 Panthera RoboTwin 原始数据到各 Policy 训练格式的转换说明，以及 ACT、Pi0.5 的训练和 ACT 评测入口。

根目录 README 只保留项目、环境和采集 Quick Start。所有 Policy 的具体转换命令、前置条件和格式约束统一记录在本文档。公共入口只负责 Policy 选择、路径适配、episode 筛选、错误处理和产物检查，转换语义尽量调用各 Policy 原生 `process_data.py`。

## 1. 基本边界

### 1.1 环境

所有原始数据转换在 `RoboTwinData` 中执行：

```bash
conda activate RoboTwinData
```

该环境不运行 SAPIEN 仿真，也不负责模型训练。ACT、Pi0、Pi0.5、RDT 等模型训练和推理需要各自的原生环境。

### 1.2 公共入口

查看某个 Policy 的参数：

```bash
python policy/data_convert.py --policy <policy> --help
```

当前注册的转换项：

```text
act  pi0  pi05  go1  rdt  tinyvla  dexvla  dp  dp3
```

输入目录通常包含：

```text
<task_config>/
├── data/episode*.hdf5
├── instructions/episode*.json
└── scene_info.json
```

公共入口支持 `--episodes` 选择非连续原始 episode，并通过临时符号链接连续编号，不复制原始 HDF5。目标目录已存在时默认拒绝覆盖，确认后才使用 `--overwrite`。

### 1.3 原始数据契约与当前限制

- RoboTwin 采集层支持单臂原生 7 维和双臂原生 14 维；
- 当前公共 Policy adapters 主要为双臂 14 维数据实现并完成 smoke，通常还假设存在 `head_camera`、`left_camera` 和 `right_camera`；
- ACT、Pi0/Pi0.5、GO1 等路径包含明确的双臂关节名、14 维 shape 或左右腕相机契约；未经单独适配和测试，不应把单臂 7 维数据直接传入；
- instruction 从每条 episode 的 `instructions/episodeN.json` 读取；
- Pi0/Pi0.5 等 LeRobot 输出必须保留 Panthera 的 motor names、robot type 和实际 FPS；
- DP3 要求原始 HDF5 的 `/pointcloud` 非空。

## 2. Policy 支持矩阵

| Policy | 转换入口 | 输出或前置条件 | 相关文档 |
| --- | --- | --- | --- |
| ACT | `policy/ACT/process_data.py` | HDF5 与 ACT 注册表匹配 | 本文 |
| Pi0 | `policy/pi0/scripts/process_data.py` | Panthera LeRobot writer | 本文 |
| Pi0.5 | `policy/pi05/scripts/process_data.py` | Panthera LeRobot writer，支持 stats/train | 本文 |
| GO1 | GO1 原生流程和 LeRobot converter | Panthera motor names、robot type、FPS | [`GO1/README.md`](GO1/README.md) |
| RDT | `policy/RDT/scripts/process_data.py` | 需要 T5 language embedding | 本文 |
| TinyVLA | `policy/TinyVLA/process_data.py` | 需要原生 task prompt | [`TinyVLA/README.md`](TinyVLA/README.md) |
| DexVLA | `policy/DexVLA/process_data.py` | 需要 prompt 和 reasoning | [`DexVLA/README.md`](DexVLA/README.md) |
| DP | `policy/DP/process_data.py` | 输出 Zarr | 本文 |
| DP3 | `policy/DP3/scripts/process_data.py` | 需要非空点云，输出 Zarr | 本文 |
| LLaVA-VLA | 暂不注册 | 原生流程要求 `front_camera` | 本文第 10 节 |

Pi0 和 Pi0.5 的第一阶段仍分别调用各自原生 `process_data.py`；第二阶段使用 Panthera LeRobot writer，因为上游 converter 面向 Aloha。两者共享同一份 Panthera writer，不复制两套实现。

## 3. ACT

### 3.1 数据转换

当前 ACT 注册表提供 `move_pillbottle_pad` 前 200 个 episode 的配置：

```bash
python policy/data_convert.py \
  --policy act \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --episodes {0..199} \
  --output data/converted/move_pillbottle_pad_panthera_act_200
```

输出包含连续编号的 `episode_*.hdf5` 和 `robotwin_conversion.json`。ACT 训练会检查：

- `dataset_dir` 指向实际转换目录；
- `num_episodes` 等于转换后的 episode 数量；
- `episode_len` 等于最大 `output_num_frames`；
- 相机顺序为 `cam_high`、`cam_right_wrist`、`cam_left_wrist`。

查看实际转换元数据：

```bash
python -c "import json; m=json.load(open('data/converted/move_pillbottle_pad_panthera_act_200/robotwin_conversion.json')); print(len(m['episodes']), max(e['output_num_frames'] for e in m['episodes']))"
```

### 3.2 训练

ACT 使用独立训练环境：

```bash
python policy/train.py \
  --policy act \
  --dataset data/converted/move_pillbottle_pad_panthera_act_200 \
  --output data/experiments/act_move_pillbottle_pad_panthera_200 \
  --epochs 2000 \
  --batch-size 1 \
  --seed 0
```

快速检查可将 `--epochs` 改为 `1`。输出至少应包含：

```text
dataset_stats.pkl
policy_best.ckpt
policy_last.ckpt
train_run.json
```

当前统一入口不支持续训或覆盖已有输出目录；请使用新的 `--output` 路径。

### 3.3 闭环评测

ACT 闭环评测需要仿真和 ACT 推理依赖：

```bash
python policy/eval.py \
  --policy act \
  --task move_pillbottle_pad \
  --task-config move_pillbottle_pad_panthera \
  --checkpoint-dir data/experiments/act_move_pillbottle_pad_panthera_200 \
  --checkpoint-name policy_best.ckpt \
  --episodes 10 \
  --gpu-id 0 \
  --fast-preview
```

结果写入 `eval_result/<task>/<policy>/<task_config>/<checkpoint>/<timestamp>/`。正式结果不要和 `--fast-preview` 的低渲染质量混合比较。

## 4. Pi0 和 Pi0.5

### 4.1 数据转换

Pi0.5：

```bash
python policy/data_convert.py \
  --policy pi05 \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --output data/converted/move_pillbottle_pad_panthera_pi05 \
  --fps 30
```

Pi0：

```bash
python policy/data_convert.py \
  --policy pi0 \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --output data/converted/move_pillbottle_pad_panthera_pi0 \
  --fps 30
```

首次验证可以追加 `--episodes 0`。当前示例输出使用 `panthera-6dof-dual`、14 维 Panthera motor names 和实际 FPS。两者均在 `RoboTwinData` 中转换，不需要完整 Pi0/OpenPI 训练环境。

### 4.2 Pi0.5 统计量和训练

Pi0.5 训练使用独立 OpenPI 环境，并使用同一个输出根目录：

```bash
policy/pi05/.venv/bin/python policy/train.py \
  --policy pi05 \
  --dataset data/converted/move_pillbottle_pad_panthera_pi05 \
  --output data/experiments/pi05_move_pillbottle_pad_panthera \
  --stage stats \
  --batch-size 64 \
  --num-workers 0
```

```bash
policy/pi05/.venv/bin/python policy/train.py \
  --policy pi05 \
  --dataset data/converted/move_pillbottle_pad_panthera_pi05 \
  --output data/experiments/pi05_move_pillbottle_pad_panthera \
  --stage train \
  --steps 20000 \
  --batch-size 64 \
  --num-workers 0
```

首次运行会下载 Pi0.5 base checkpoint，需要网络、缓存、磁盘和足够显存。显存不足时先降低 batch size；不要未经验证就假设梯度累积等价于更大的 batch size。

## 5. GO1

```bash
python policy/data_convert.py \
  --policy go1 \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --output data/converted/move_pillbottle_pad_panthera_go1 \
  --fps 30
```

公共入口只给原生 GO1 writer 传入 Panthera motor names、robot type 和 FPS，并兼容固定 LeRobot 版本的 `add_frame(frame)` API。GO1 模型训练按 [`GO1/README.md`](GO1/README.md) 使用独立环境。

## 6. DP 和 DP3

### 6.1 DP

```bash
python policy/data_convert.py \
  --policy dp \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --output data/converted/move_pillbottle_pad_panthera_dp.zarr
```

### 6.2 DP3

```bash
python policy/data_convert.py \
  --policy dp3 \
  --input <使用 pointcloud=true 采集的数据目录> \
  --output data/converted/panthera_dp3.zarr
```

DP3 要求每条原始 HDF5 的 `/pointcloud` 非空。只有 key 但 shape 为 `(T, 0)` 不能作为 DP3 输入。DP 和 DP3 原生流程可能累积全部 episode 到内存，大规模转换前应做压力测试。

## 7. RDT

```bash
python policy/data_convert.py \
  --policy rdt \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --output data/converted/move_pillbottle_pad_panthera_rdt \
  --gpu-id 0
```

RDT 需要完整的 Hugging Face `google/t5-v1_1-xxl` 快照：

```bash
hf download \
  google/t5-v1_1-xxl \
  --local-dir policy/weights/RDT/t5-v1_1-xxl
```

需要 tokenizer、配置和 T5 encoder 权重；不是 RDT-1B checkpoint，也不是 SigLIP 图像编码器。原生 encoder 默认倾向使用 GPU，16 GB 显存可能 OOM。

## 8. TinyVLA 和 DexVLA

这两个 Policy 的原生 `process_data.py` 要求任务 prompt 预先定义在源码字典中，公共入口不会自动覆盖。

TinyVLA 示例：

```python
# policy/TinyVLA/process_data.py
task_prompt = {
    "move_pillbottle_pad": "Pick up the pill bottle and place it on the pad.",
}
```

```bash
python policy/data_convert.py \
  --policy tinyvla \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --output data/converted/move_pillbottle_pad_panthera_tinyvla \
  --task-name move_pillbottle_pad
```

DexVLA 还需要 reasoning：

```python
# policy/DexVLA/process_data.py
task_prompt = {
    "move_pillbottle_pad": "Pick up the pill bottle and place it on the pad.",
}
task_reasoning = {
    "move_pillbottle_pad": 0,
}
all_reasoning = [
    ["Locate the bottle, grasp it, move it above the pad, and release it."],
]
```

```bash
python policy/data_convert.py \
  --policy dexvla \
  --input data/move_pillbottle_pad/move_pillbottle_pad_panthera \
  --output data/converted/move_pillbottle_pad_panthera_dexvla \
  --task-name move_pillbottle_pad
```

## 9. 全量转换和 instruction 注意事项

- GO1 的 LeRobot `save_episode()` 可能随 episode 数增加内存；全量转换前应压力测试。
- DP/DP3 原生脚本可能累积所有图像、状态、动作或点云；先用小子集验证峰值内存。
- RoboTwin instruction 是采集完成后从 `scene_info.json` 派生的。原生生成器会在模板不足时循环补齐 `language_num`，所以 nominal 数量不等于唯一语言数量；修复生成器后只需重生成 instruction JSON。
- Pi0/Pi0.5、RDT、GO1、TinyVLA 和 DexVLA 等语言条件链路会读取 instruction；重复或生硬措辞可能降低有效语言多样性，但不改变已经采集的动作、图像和物理轨迹。

## 10. 暂不支持 LLaVA-VLA

LLaVA-VLA 原生图像流程固定读取 `front_camera`，而当前 Panthera 原始数据没有该相机。本仓库不伪造或复制相机数据，因此没有将其注册到公共转换入口。
