# RoboTwin Panthera

本仓库基于 [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)，面向 Panthera 六自由度机械臂，提供仿真、专家数据采集、轨迹回放和多 Policy 数据转换能力。

当前适配支持：

- 双臂 Panthera：左右两台机械臂，原生 14 维动作；
- 单臂 Panthera：场景只初始化一台居中的机械臂，原生 7 维动作（6 个关节 + 1 个夹爪）；
- RoboTwin 原生 CuRobo、TOPP、SAPIEN 执行和场景随机化接口；
- 头部相机、腕部相机、HDF5、轨迹、视频和物理准入报告输出；
- 通过 [`RoboTwin/policy/README.md`](RoboTwin/policy/README.md) 转换 ACT、Pi0、Pi0.5、GO1、RDT、TinyVLA、DexVLA、DP 和 DP3 数据。

本仓库不是 RoboTwin 官方发行版。当前文档只承诺下方明确列出的流程；其他任务、随机化组合和真机验证仍需逐项审查。

## 项目状态

| 能力 | 状态 |
| --- | --- |
| `move_pillbottle_pad` 双臂采集 | 已验证 |
| `move_pillbottle_pad` 单臂采集 | 已验证，10 条正式 episode |
| `blocks_ranking_rgb` 双臂配置与回归 | 已完成配置和真实回归 |
| `blocks_ranking_rgb` 单臂采集 | 已验证，10 条正式 episode、30 次方块抓放周期 |
| 单臂/双臂原始轨迹回放 | 已验证 |
| 多 Policy 数据转换 | 已接入；当前 adapters 主要面向双臂 14 维数据，具体边界见 [`RoboTwin/policy/README.md`](RoboTwin/policy/README.md) |
| ACT 训练与评测 | 有统一入口，见 [`RoboTwin/policy/README.md`](RoboTwin/policy/README.md) |
| Pi0.5 统计量计算与训练 | 有统一入口，见 [`RoboTwin/policy/README.md`](RoboTwin/policy/README.md) |
| 所有 RoboTwin 任务的单臂适配 | 未宣称完成，必须逐任务审查 |
| Panthera 真机控制与 sim-to-real 安全保证 | 不在本仓库范围内 |

## 仓库目录

```text
RoboTwin/
├── assets/
│   └── embodiments/panthera-6dof/   # Panthera URDF、SRDF、网格和 CuRobo 配置
├── description/                      # 任务和物体语言模板、instruction 生成器
├── envs/                             # RoboTwin 任务、机器人和相机仿真逻辑
├── task_config/                      # 任务、相机、embodiment 和采集配置
├── script/
│   ├── collect_data.py               # 采集主入口
│   ├── replay_data.py                # SAPIEN Viewer 密集轨迹回放
│   ├── eval_policy.py                # RoboTwin 原生评测入口
│   └── _download_assets.sh           # RoboTwin 通用资产下载脚本
├── policy/
│   ├── README.md                     # Policy 转换、训练和评测说明
│   ├── data_convert.py               # 公共 Policy 数据转换入口
│   ├── train.py                      # ACT/Pi0.5 统一训练入口
│   ├── eval.py                       # ACT 统一评测入口
│   └── <policy>/                     # 各 Policy 原生代码
└── README.md
```

## 1. 环境配置

建议使用 Linux、NVIDIA GPU、Conda 和可用的 Vulkan。采集和数据转换必须使用不同环境：

| 环境 | 用途 | 主要版本 |
| --- | --- | --- |
| `RoboTwin` | Panthera 仿真、CuRobo/TOPP 规划、SAPIEN 采集、视频和 HDF5 | Python 3.10.20，PyTorch 2.7.1 + CUDA 12.8，SAPIEN 3.0.0b1，MPlib 0.2.1，CuRobo 0.7.8 |
| `RoboTwinData` | 原始 HDF5 到各 Policy 训练格式的转换 | Python 3.11.15，PyTorch 2.6.0 + CUDA 12.4，Zarr 2.18.4，固定 LeRobot commit |

Policy 模型训练和推理环境不属于这两个基础环境，必须按对应 Policy 的原生要求单独配置。

### 1.1 主机依赖

```bash
sudo apt update
sudo apt install -y git unzip libvulkan1 mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary
```

还需要正确安装 NVIDIA 驱动，并确认 Vulkan 使用 NVIDIA ICD，而不是 CPU 渲染设备。

### 1.2 `RoboTwin` 采集环境

```bash
conda create -n RoboTwin python=3.10.20 -y
conda activate RoboTwin
conda install -y -c nvidia/label/cuda-12.8.0 cuda-compiler=12.8
conda install -y -c conda-forge ffmpeg=7

python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  numpy==1.26.4 scipy==1.10.1 sapien==3.0.0b1 mplib==0.2.1 \
  transforms3d==0.4.2 trimesh==4.4.3 open3d==0.18.0 \
  gymnasium==0.29.1 toppra==0.6.9 h5py==3.16.0 \
  opencv-python==4.11.0.86 imageio==2.34.2 imageio-ffmpeg==0.6.0 \
  Pillow==11.3.0 PyYAML==6.0.3 huggingface-hub==0.25.0 \
  ninja==1.13.0 wheel==0.47.0 warp-lang==1.12.0

python -m pip install \
  "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" \
  --no-build-isolation
git clone --branch v0.7.8 --depth 1 \
  https://github.com/NVlabs/curobo.git envs/curobo
python -m pip install -e envs/curobo --no-build-isolation
```

当前仓库还包含针对 RoboTwin 2.0、SAPIEN SRDF 查找和 MPlib planner 失败判定的兼容性处理。不要把上游 `bash script/_install.sh` 直接当作当前环境的完整安装方案；公开发布前仍需在干净机器上做一次 clean-room 验收。

检查核心依赖：

```bash
python -c "import torch, sapien, mplib, curobo, pytorch3d; print(torch.__version__, torch.version.cuda)"
ffmpeg -version
python -m pip check
```

### 1.3 `RoboTwinData` 转换环境

```bash
conda create -n RoboTwinData python=3.11.15 -y
conda activate RoboTwinData
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  numpy==1.26.4 h5py==3.16.0 opencv-python==4.11.0.86 \
  PyYAML==6.0.3 Pillow==11.0.0 datasets==3.2.0 \
  pyarrow==18.1.0 tqdm==4.67.1 zarr==2.18.4 numcodecs==0.13.1 \
  huggingface-hub==0.28.1 diffusers==0.31.0 rerun-sdk==0.21.0 \
  transformers==4.41.0 sentencepiece==0.2.0 accelerate==0.30.1
python -m pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
```

Zarr 必须保持 2.x；LeRobot 必须固定到已验证的 commit。检查：

```bash
python -c "import torch, h5py, cv2, zarr, lerobot, transformers; print(torch.__version__, zarr.__version__)"
python policy/data_convert.py --policy act --help
python -m pip check
```

## 2. 准备仿真资产

Panthera 资产随本仓库发布；RoboTwin 通用任务物体、背景纹理和其他通用 embodiment 通过下载脚本获取：

```bash
bash script/_download_assets.sh
cd assets
unzip background_texture.zip
unzip embodiments.zip
unzip objects.zip
cd ..
```

如果 Hugging Face 限流，先执行 `huggingface-cli login`。Panthera 目录至少应包含：

```text
assets/embodiments/panthera-6dof/
├── panthera_6dof.urdf
├── panthera_6dof.srdf
├── config.yml
├── curobo.yml
├── collision_panthera.yml
└── meshes/
```

检查资产：

```bash
test -f assets/embodiments/panthera-6dof/panthera_6dof.urdf
test -f assets/embodiments/panthera-6dof/panthera_6dof.srdf
test -f assets/embodiments/panthera-6dof/config.yml
test -f assets/embodiments/panthera-6dof/curobo.yml
test -d assets/embodiments/panthera-6dof/meshes
test -d assets/objects
test -d assets/background_texture
```

## 3. Quick Start

以下命令均从仓库根目录执行。

### 3.1 采集双臂数据

先激活 `RoboTwin` 环境，并把配置中的 `episode_num` 临时设为 `1` 做 smoke：

```bash
conda activate RoboTwin
bash collect_data.sh move_pillbottle_pad move_pillbottle_pad_panthera 0
```

### 3.2 采集单臂数据

单臂配置只初始化一台居中 Panthera，输出原生 7 维数据：

```bash
bash collect_data.sh move_pillbottle_pad move_pillbottle_pad_panthera_single 0
bash collect_data.sh blocks_ranking_rgb blocks_ranking_rgb_panthera_single 0
```

`blocks_ranking_rgb` 当前也提供双臂配置：

```bash
bash collect_data.sh blocks_ranking_rgb blocks_ranking_rgb_panthera 0
```

采集结果位于：

```text
data/<task>/<task_config>/
├── data/episodeN.hdf5
├── _traj_data/episodeN.pkl
├── instructions/episodeN.json
├── physics_validation/episodeN_physics.json
├── video/
├── scene_info.json
└── seed.txt
```

单臂 HDF5 的动作形状是 `arm=[T,6]`、`gripper=[T]`、`vector=[T,7]`；双臂为 14 维。采集完成后，instruction JSON 由 `scene_info.json` 派生生成，不参与专家轨迹规划。

### 3.3 回放一条采集轨迹

```bash
python script/replay_data.py \
  data/blocks_ranking_rgb/blocks_ranking_rgb_panthera_single/data/episode0.hdf5
```

双臂示例：

```bash
python script/replay_data.py \
  data/blocks_ranking_rgb/blocks_ranking_rgb_panthera/data/episode0.hdf5
```

回放要求保留同一 episode 的 HDF5、`_traj_data/episodeN.pkl` 和 `seed.txt`。

### 3.4 转换 Policy 数据

Policy 转换全部在 `RoboTwinData` 中执行，具体命令见 [`RoboTwin/policy/README.md`](RoboTwin/policy/README.md)：

```bash
conda activate RoboTwinData
python policy/data_convert.py --policy <act|pi0|pi05|go1|rdt|tinyvla|dexvla|dp|dp3> --help
```

当前公共 adapters 主要按双臂 14 维状态/动作和头部、左右腕部三路相机实现。单臂 7 维数据已经完成采集验证，但尚不能直接推断所有 Policy converter 都已支持该 schema。

## 4. Common Issues

### 环境和渲染

- 采集必须在 `RoboTwin` 中执行；Policy 数据转换必须在 `RoboTwinData` 中执行；模型训练和推理使用对应 Policy 的独立环境。
- SAPIEN 或 Vulkan 报错时，检查 NVIDIA 驱动、Vulkan ICD 和 Panthera 完整资产，而不是只检查 URDF。
- `ffmpeg` 不存在时视频写入会失败；采集环境应安装真正的 `ffmpeg` 命令行程序。
- RTX 5060 Ti 可能出现 OIDN CUDA 兼容警告。只要 HDF5 RGB 非空、视频完整解码且帧数对齐，该警告本身不等于数据损坏。

### 采集和回放

- 当前配置文件使用 `.yml`，例如 `blocks_ranking_rgb_panthera.yml`，不是 `.yaml`。
- 单臂和双臂不能混用轨迹、seed 或 HDF5；回放会校验 schema、动作维度和轨迹模式。
- 采集失败时先查看 `physics_validation/` 报告。正式 HDF5 只在规划、任务成功和物理准入全部通过后合并。
- `DP3` 要求原始 HDF5 含非空 `/pointcloud`；默认 RGB 采集配置通常不会满足这一条件。

### Instruction 和 Policy

- RoboTwin 原生 instruction 生成器会在模板不足时循环补齐 `language_num`，因此 nominal 数量不等于唯一语言数量。
- instruction 是采集结束后根据 `scene_info.json` 派生的。修复生成器后可以只重生成 JSON，不必重采集视频和物理轨迹。
- 具体 Policy 的转换、训练、权重和前置条件统一查看 [`RoboTwin/policy/README.md`](RoboTwin/policy/README.md)。

### 发布前检查

- 当前环境依赖尚未在全新机器上完成 clean-room 复现；公开发布前应重新验证安装和最小采集 smoke。
- 当前只对已列出的 Panthera 任务做过真实验证，不能据此推断所有 RoboTwin 任务都支持单臂。
- 本仓库不包含 Panthera 真机 SDK、真机控制或 sim-to-real 安全保证。

## License

除另有说明外，仓库外层新增文件由 HighTorque Robotics 按 [MIT License](LICENSE) 发布。`RoboTwin/` 目录保留 RoboTwin 原有许可证、版权声明及第三方组件许可证；其上游 README、论文和 BibTeX 引用见 [`RoboTwin/docs/ROBOTWIN_UPSTREAM_README.md`](RoboTwin/docs/ROBOTWIN_UPSTREAM_README.md)。

`policy/` 下的第三方项目可能带有各自的许可证，使用和再分发时必须同时遵守对应条款。Panthera URDF、网格和配套配置由本项目提供并随仓库公开发布。
