"""Convert RoboTwin raw episodes into policy datasets.

The public boundary in this module is the RoboTwin raw HDF5 format. Policy
adapters own all policy-specific feature names and preprocessing.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Protocol

import h5py


class ConversionError(RuntimeError):
    """Base class for actionable conversion failures."""


class InputError(ConversionError):
    pass


class AdapterError(ConversionError):
    pass


CAMERA_MAPPING = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

PANTHERA_MOTOR_NAMES = [
    *(f"left_joint_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(6)),
    "right_gripper",
]


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    """Discovered episode path; payload is loaded only during iteration."""

    episode_id: int
    source_path: Path


@dataclasses.dataclass
class RobotTwinDataset:
    input_dir: Path
    episodes: list[EpisodeRef]

    @classmethod
    def load(
        cls,
        input_dir: Path,
        episode_ids: list[int] | None = None,
    ) -> "RobotTwinDataset":
        if not input_dir.is_dir():
            raise InputError(f"输入目录不存在: {input_dir}")
        data_dir = input_dir / "data"
        if not data_dir.is_dir():
            raise InputError(f"输入目录缺少 data/: {data_dir}")
        files = sorted(data_dir.glob("episode*.hdf5"), key=_episode_number)
        if not files:
            raise InputError(f"未发现 data/episode*.hdf5: {data_dir}")
        selected = set(episode_ids) if episode_ids is not None else None
        episodes: list[EpisodeRef] = []
        for path in files:
            episode_id = _episode_number(path)
            if selected is not None and episode_id not in selected:
                continue
            episodes.append(EpisodeRef(episode_id, path))
        if selected is not None:
            found = {ep.episode_id for ep in episodes}
            missing = sorted(selected - found)
            if missing:
                raise InputError(f"指定 episode 不存在: {missing}")
        if not episodes:
            raise InputError("没有选中的 episode")
        return cls(
            input_dir=input_dir,
            episodes=episodes,
        )


class PolicyAdapter(Protocol):
    name: str
    description: str

    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    def run(self, args: argparse.Namespace) -> None: ...


def _episode_number(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.removeprefix("episode"))
    except ValueError as exc:
        raise InputError(f"非法 episode 文件名: {path.name}") from exc


def _add_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_fps: bool = False,
    require_fps: bool = False,
) -> None:
    parser.add_argument("--input", type=Path, required=True, help="RoboTwin 任务采集目录")
    parser.add_argument("--output", type=Path, required=True, help="最终 Policy 数据集目录")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="可选的原始 episode ID")
    if include_fps or require_fps:
        parser.add_argument(
            "--fps",
            type=float,
            required=require_fps,
            help="数据帧率" if require_fps else "可选的数据帧率，仅写入转换清单",
        )
    parser.add_argument("--overwrite", action="store_true", help="显式覆盖已有输出目录")


def _run_atomic_dataset_conversion(
    adapter: PolicyAdapter,
    args: argparse.Namespace,
    convert: Callable[[RobotTwinDataset, Path, argparse.Namespace], None],
) -> None:
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise InputError(f"输出目录已存在，若要覆盖请显式添加 --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = RobotTwinDataset.load(
        args.input.resolve(),
        args.episodes,
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        convert(dataset, staging, args)
        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"转换完成: policy={adapter.name}, episodes={len(dataset.episodes)}, output={output}",
        flush=True,
    )


def _instruction_path(dataset: RobotTwinDataset, episode: EpisodeRef) -> Path:
    path = dataset.input_dir / "instructions" / f"episode{episode.episode_id}.json"
    if not path.is_file():
        raise InputError(f"原始 episode 缺少 instruction: {path}")
    return path


def _link_episodes(
    dataset: RobotTwinDataset,
    raw_root: Path,
    *,
    nested_data: bool,
    include_instructions: bool,
) -> None:
    data_root = raw_root / "data" if nested_data else raw_root
    data_root.mkdir(parents=True, exist_ok=True)
    if include_instructions:
        (raw_root / "instructions").mkdir(parents=True, exist_ok=True)
    for output_episode_id, episode in enumerate(dataset.episodes):
        (data_root / f"episode{output_episode_id}.hdf5").symlink_to(
            episode.source_path.resolve()
        )
        if include_instructions:
            instruction_link = (
                raw_root / "instructions" / f"episode{output_episode_id}.json"
            )
            instruction_link.symlink_to(_instruction_path(dataset, episode).resolve())


def _run_native_script(
    script: Path,
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    try:
        subprocess.run(
            [sys.executable, str(script.resolve()), *arguments],
            cwd=cwd,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AdapterError(
            f"原生转换脚本执行失败: {script}: exit={exc.returncode}"
        ) from exc


class OpenPiPantheraAdapter:
    """Shared orchestration for Pi0/Pi0.5; policy conversion stays native."""

    name: str
    description: str
    process_data_path: Path

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_dataset_arguments(parser, require_fps=True)

    def run(self, args: argparse.Namespace) -> None:
        _run_atomic_dataset_conversion(self, args, self.convert)

    def convert(
        self,
        dataset: RobotTwinDataset,
        output: Path,
        args: argparse.Namespace,
    ) -> None:
        if args.fps <= 0:
            raise AdapterError(f"{self.name} 转换必须显式提供正数 --fps")

        process_data = _load_module(
            f"robotwin_{self.name}_process_data",
            self.process_data_path,
        )
        lerobot_converter = _load_module(
            "robotwin_pi05_lerobot_converter",
            Path(__file__).parent
            / "pi05"
            / "scripts"
            / "convert_robotwin_to_lerobot.py",
        )

        with tempfile.TemporaryDirectory(prefix=f"robotwin_{self.name}_") as temp_dir:
            temp_root = Path(temp_dir)
            raw_root = temp_root / "raw"
            (raw_root / "data").mkdir(parents=True)
            (raw_root / "instructions").mkdir()
            writer = None
            try:
                for index, episode in enumerate(dataset.episodes):
                    raw_path = raw_root / "data" / "episode0.hdf5"
                    raw_path.symlink_to(episode.source_path.resolve())
                    instruction_path = raw_root / "instructions" / "episode0.json"
                    instruction_path.symlink_to(
                        _instruction_path(dataset, episode).resolve()
                    )
                    processed_root = temp_root / f"processed_{index}"
                    process_data.data_transform(str(raw_root), 1, str(processed_root))
                    processed_file = processed_root / "episode_0" / "episode_0.hdf5"
                    if writer is None:
                        writer = lerobot_converter.LeRobotDatasetWriter(
                            output_root=output.parent,
                            repo_id=output.name,
                            first_episode=processed_file,
                            fps=args.fps,
                            robot_type="panthera-6dof-dual",
                        )
                    writer.add_episode(processed_file)
                    shutil.rmtree(processed_root)
                    raw_path.unlink()
                    instruction_path.unlink()
            finally:
                if writer is not None:
                    writer.close()

        (output / "robotwin_conversion.json").write_text(
            json.dumps(
                {
                    "policy": self.name,
                    "robot_type": "panthera-6dof-dual",
                    "action_alignment": "next_state",
                    "state_semantics": "state_target",
                    "fps": args.fps,
                    "camera_mapping": CAMERA_MAPPING,
                    "adapter": (
                        f"{self.process_data_path.relative_to(Path(__file__).parent.parent)}"
                        " + policy/pi05/scripts/convert_robotwin_to_lerobot.py"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class Pi0Adapter(OpenPiPantheraAdapter):
    name = "pi0"
    description = "使用 Pi0 原生 process_data 和 Panthera LeRobot writer 转换数据"
    process_data_path = Path(__file__).parent / "pi0" / "scripts" / "process_data.py"


class Pi05Adapter(OpenPiPantheraAdapter):
    name = "pi05"
    description = "使用 Pi0.5 原生 process_data 和 Panthera LeRobot writer 转换数据"
    process_data_path = Path(__file__).parent / "pi05" / "scripts" / "process_data.py"


class Go1Adapter:
    name = "go1"
    description = "调用 GO1 原生两阶段转换，输出 Panthera LeRobot Dataset"
    policy_root = Path(__file__).parent / "GO1"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_dataset_arguments(parser)
        parser.add_argument("--fps", type=int, required=True, help="GO1 LeRobot Dataset 帧率")

    def run(self, args: argparse.Namespace) -> None:
        _run_atomic_dataset_conversion(self, args, self.convert)

    def convert(
        self,
        dataset: RobotTwinDataset,
        output: Path,
        args: argparse.Namespace,
    ) -> None:
        if args.fps <= 0:
            raise InputError("GO1 转换要求 --fps 为正整数")
        process_data = _load_module(
            "robotwin_go1_process_data",
            self.policy_root / "scripts" / "process_data.py",
        )

        with tempfile.TemporaryDirectory(prefix="robotwin_go1_") as temp_dir:
            temp_root = Path(temp_dir)
            raw_root = temp_root / "raw"
            _link_episodes(
                dataset,
                raw_root,
                nested_data=True,
                include_instructions=True,
            )
            processed_root = temp_root / "processed"
            process_data.data_transform(
                str(raw_root),
                len(dataset.episodes),
                str(processed_root),
            )

            old_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
            os.environ["HF_LEROBOT_HOME"] = str(output.parent)
            try:
                converter = _load_module(
                    "robotwin_go1_lerobot_converter",
                    self.policy_root
                    / "scripts"
                    / "convert_aloha_data_to_lerobot_robotwin.py",
                )
                converter.port_aloha(
                    raw_dir=processed_root,
                    repo_id=output.name,
                    mode="image",
                    fps=args.fps,
                    robot_type="panthera-6dof-dual",
                    motor_names=PANTHERA_MOTOR_NAMES,
                )
            finally:
                if old_lerobot_home is None:
                    os.environ.pop("HF_LEROBOT_HOME", None)
                else:
                    os.environ["HF_LEROBOT_HOME"] = old_lerobot_home

        if not output.is_dir():
            raise AdapterError(f"GO1 原生转换没有生成预期 LeRobot Dataset: {output}")
        info_path = output / "meta" / "info.json"
        if not info_path.is_file():
            raise AdapterError(f"GO1 LeRobot Dataset 缺少元数据: {info_path}")
        (output / "robotwin_conversion.json").write_text(
            json.dumps(
                {
                    "policy": "go1",
                    "format": "lerobot",
                    "robot_type": "panthera-6dof-dual",
                    "motor_names": PANTHERA_MOTOR_NAMES,
                    "fps": args.fps,
                    "action_alignment": "next_state",
                    "state_semantics": "state_target",
                    "adapter": "policy/GO1/scripts/process_data.py + "
                    "policy/GO1/scripts/convert_aloha_data_to_lerobot_robotwin.py",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class ActAdapter:
    """Dispatch selected episodes to ACT's original data_transform entry."""

    name = "act"
    description = "使用 ACT 原生 process_data 转换 RoboTwin HDF5"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_dataset_arguments(parser, include_fps=True)

    def run(self, args: argparse.Namespace) -> None:
        _run_atomic_dataset_conversion(self, args, self.convert)

    def convert(
        self,
        dataset: RobotTwinDataset,
        output: Path,
        args: argparse.Namespace,
    ) -> None:
        process_data = _load_module(
            "robotwin_act_process_data",
            Path(__file__).parent / "ACT" / "process_data.py",
        )
        with tempfile.TemporaryDirectory(prefix="robotwin_act_") as temp_dir:
            input_dir = Path(temp_dir)
            for output_episode_id, episode in enumerate(dataset.episodes):
                (input_dir / f"episode{output_episode_id}.hdf5").symlink_to(
                    episode.source_path.resolve()
                )
            process_data.data_transform(
                str(input_dir),
                len(dataset.episodes),
                str(output),
            )

        converted_episodes = []
        for output_episode_id, episode in enumerate(dataset.episodes):
            episode_path = output / f"episode_{output_episode_id}.hdf5"
            with h5py.File(episode_path, "r") as root:
                action_shape = root["action"].shape
                qpos_shape = root["observations/qpos"].shape
            if (
                action_shape != qpos_shape
                or len(action_shape) != 2
                or action_shape[1] != 14
            ):
                raise AdapterError(
                    f"ACT 转换结果维度错误: {episode_path}: "
                    f"action={action_shape}, qpos={qpos_shape}"
                )
            converted_episodes.append(
                {
                    "source_episode_id": episode.episode_id,
                    "output_episode_id": output_episode_id,
                    "source_num_frames": action_shape[0] + 1,
                    "output_num_frames": action_shape[0],
                }
            )

        (output / "robotwin_conversion.json").write_text(
            json.dumps(
                {
                    "policy": "act",
                    "format": "act-hdf5",
                    "robot_type": "panthera-6dof-dual",
                    "action_dim": 14,
                    "action_alignment": "next_state",
                    "state_semantics": "state_target",
                    "camera_mapping": CAMERA_MAPPING,
                    "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
                    "image_size": [480, 640],
                    "fps": args.fps if args.fps and args.fps > 0 else None,
                    "episodes": converted_episodes,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class NativeZarrAdapter:
    """Run a Policy's original CLI converter in an isolated directory."""

    name: str
    description: str
    script_path: Path
    requires_pointcloud = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_dataset_arguments(parser)

    def run(self, args: argparse.Namespace) -> None:
        _run_atomic_dataset_conversion(self, args, self.convert)

    def convert(
        self,
        dataset: RobotTwinDataset,
        output: Path,
        args: argparse.Namespace,
    ) -> None:
        for episode in dataset.episodes:
            with h5py.File(episode.source_path, "r") as root:
                if "/joint_action/vector" not in root:
                    raise InputError(f"{episode.source_path}: 缺少 /joint_action/vector")
                if self.requires_pointcloud:
                    if "/pointcloud" not in root:
                        raise InputError(f"{episode.source_path}: DP3 转换要求 /pointcloud")
                    pointcloud_shape = root["/pointcloud"].shape
                    if len(pointcloud_shape) < 2 or pointcloud_shape[1] == 0:
                        raise InputError(
                            f"{episode.source_path}: /pointcloud 为空，shape={pointcloud_shape}; "
                            "请先使用启用 pointcloud 的任务配置重新采集"
                        )

        task_name = "robotwin_data_convert"
        task_config = "selected"
        with tempfile.TemporaryDirectory(
            prefix=f"robotwin_{self.name}_native-",
            dir=output.parent,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            work_dir = temp_root / "policy" / self.name
            work_dir.mkdir(parents=True)
            raw_root = temp_root / "data" / task_name / task_config
            _link_episodes(
                dataset,
                raw_root,
                nested_data=True,
                include_instructions=False,
            )
            _run_native_script(
                self.script_path,
                [task_name, task_config, str(len(dataset.episodes))],
                cwd=work_dir,
            )
            native_output = (
                work_dir
                / "data"
                / f"{task_name}-{task_config}-{len(dataset.episodes)}.zarr"
            )
            if not native_output.is_dir():
                raise AdapterError(
                    f"原生 {self.name} 转换没有生成预期 Zarr: {native_output}"
                )
            if output.exists():
                output.rmdir()
            native_output.replace(output)

        required = [
            output / "data" / "state" / ".zarray",
            output / "data" / "action" / ".zarray",
            output / "meta" / "episode_ends" / ".zarray",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise AdapterError(f"{self.name} Zarr 缺少必要字段: {missing}")


class DpAdapter(NativeZarrAdapter):
    name = "dp"
    description = "运行 DP 原生 process_data.py，生成最终 Zarr"
    script_path = Path(__file__).parent / "DP" / "process_data.py"


class Dp3Adapter(NativeZarrAdapter):
    name = "dp3"
    description = "运行 DP3 原生 process_data.py，生成最终点云 Zarr"
    script_path = Path(__file__).parent / "DP3" / "scripts" / "process_data.py"
    requires_pointcloud = True


class NativeVlaHdf5Adapter:
    """Call TinyVLA/DexVLA's own HDF5 transformer without reimplementing it."""

    name: str
    description: str
    process_data_path: Path
    requires_reasoning = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_dataset_arguments(parser)
        parser.add_argument(
            "--task-name",
            required=True,
            help="必须已在该 Policy 原生 process_data.py 的提示词字典中定义",
        )

    def run(self, args: argparse.Namespace) -> None:
        _run_atomic_dataset_conversion(self, args, self.convert)

    def convert(
        self,
        dataset: RobotTwinDataset,
        output: Path,
        args: argparse.Namespace,
    ) -> None:
        process_data = _load_module(
            f"robotwin_{self.name}_process_data",
            self.process_data_path,
        )
        if args.task_name not in process_data.task_prompt:
            raise InputError(
                f"{self.name} 原生 process_data.py 未定义任务提示词: {args.task_name!r}; "
                "请先按 README 示例补充 task_prompt"
            )
        if self.requires_reasoning:
            if args.task_name not in process_data.task_reasoning:
                raise InputError(
                    f"{self.name} 原生 process_data.py 未定义任务 reasoning: {args.task_name!r}; "
                    "请先按 README 示例补充 task_reasoning 和 all_reasoning"
                )
            reasoning_index = process_data.task_reasoning[args.task_name]
            if not isinstance(reasoning_index, int) or not 0 <= reasoning_index < len(
                process_data.all_reasoning
            ):
                raise InputError(
                    f"{self.name} 的 task_reasoning[{args.task_name!r}] 索引无效: {reasoning_index!r}"
                )

        with tempfile.TemporaryDirectory(prefix=f"robotwin_{self.name}_") as temp_dir:
            input_dir = Path(temp_dir)
            _link_episodes(
                dataset,
                input_dir,
                nested_data=False,
                include_instructions=False,
            )
            process_data.data_transform(
                str(input_dir),
                len(dataset.episodes),
                str(output),
                args.task_name,
            )

        for output_episode_id in range(len(dataset.episodes)):
            episode_path = output / f"episode_{output_episode_id}.hdf5"
            if not episode_path.is_file():
                raise AdapterError(f"{self.name} 没有生成预期文件: {episode_path}")
            with h5py.File(episode_path, "r") as root:
                required = [
                    "action",
                    "language_raw",
                    "observations/qpos",
                    "observations/images",
                ]
                if self.requires_reasoning:
                    required.append("reasoning")
                missing = [key for key in required if key not in root]
                if missing:
                    raise AdapterError(f"{episode_path}: 缺少原生训练字段 {missing}")
                if root["action"].shape != root["observations/qpos"].shape:
                    raise AdapterError(f"{episode_path}: action/qpos shape 不一致")

        (output / "robotwin_conversion.json").write_text(
            json.dumps(
                {
                    "policy": self.name,
                    "format": "policy-native-hdf5",
                    "robot_type": "panthera-6dof-dual",
                    "task_name": args.task_name,
                    "action_alignment": "next_state",
                    "state_semantics": "state_target",
                    "adapter": str(
                        self.process_data_path.relative_to(Path(__file__).parent.parent)
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class TinyVlaAdapter(NativeVlaHdf5Adapter):
    name = "tinyvla"
    description = "调用 TinyVLA 原生 process_data.py，生成最终训练 HDF5"
    process_data_path = Path(__file__).parent / "TinyVLA" / "process_data.py"


class DexVlaAdapter(NativeVlaHdf5Adapter):
    name = "dexvla"
    description = "调用 DexVLA 原生 process_data.py，生成含 reasoning 的最终训练 HDF5"
    process_data_path = Path(__file__).parent / "DexVLA" / "process_data.py"
    requires_reasoning = True


class RdtAdapter:
    name = "rdt"
    description = "运行 RDT 原生 process_data.py 和 T5 语言编码，生成最终训练目录"
    policy_root = Path(__file__).parent / "RDT"
    script_path = policy_root / "scripts" / "process_data.py"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_dataset_arguments(parser)
        parser.add_argument(
            "--gpu-id",
            type=int,
            default=0,
            help="RDT T5 语言编码使用的 GPU",
        )

    def run(self, args: argparse.Namespace) -> None:
        _run_atomic_dataset_conversion(self, args, self.convert)

    def convert(
        self,
        dataset: RobotTwinDataset,
        output: Path,
        args: argparse.Namespace,
    ) -> None:
        weights = Path(__file__).parent / "weights" / "RDT" / "t5-v1_1-xxl"
        if not weights.is_dir():
            raise InputError(f"RDT 语言编码缺少 T5 权重目录: {weights}")
        for episode in dataset.episodes:
            _instruction_path(dataset, episode)

        task_name = "robotwin_data_convert"
        task_config = "selected"
        with tempfile.TemporaryDirectory(
            prefix="robotwin_rdt_native-",
            dir=output.parent,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            work_dir = temp_root / "policy" / "RDT"
            work_dir.mkdir(parents=True)
            (work_dir / "scripts").symlink_to((self.policy_root / "scripts").resolve())
            (work_dir / "models").symlink_to((self.policy_root / "models").resolve())
            (work_dir / "configs").symlink_to((self.policy_root / "configs").resolve())
            (temp_root / "policy" / "weights").symlink_to(
                (Path(__file__).parent / "weights").resolve()
            )
            raw_root = temp_root / "data" / task_name / task_config
            _link_episodes(
                dataset,
                raw_root,
                nested_data=True,
                include_instructions=True,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            _run_native_script(
                self.script_path,
                [task_name, task_config, str(len(dataset.episodes))],
                cwd=work_dir,
                env=env,
            )
            native_output = (
                work_dir
                / "processed_data"
                / f"{task_name}-{task_config}-{len(dataset.episodes)}"
            )
            if not native_output.is_dir():
                raise AdapterError(f"RDT 原生转换没有生成预期目录: {native_output}")
            if output.exists():
                output.rmdir()
            native_output.replace(output)

        for output_episode_id in range(len(dataset.episodes)):
            episode_root = output / f"episode_{output_episode_id}"
            if not (episode_root / f"episode_{output_episode_id}.hdf5").is_file():
                raise AdapterError(f"RDT 缺少转换后 HDF5: {episode_root}")
            embeddings = episode_root / "instructions"
            if not embeddings.is_dir() or not any(embeddings.glob("lang_embed_*.pt")):
                raise AdapterError(f"RDT 缺少语言 embedding: {embeddings}")

        (output / "robotwin_conversion.json").write_text(
            json.dumps(
                {
                    "policy": "rdt",
                    "format": "rdt-native-hdf5-with-language-embeddings",
                    "robot_type": "panthera-6dof-dual",
                    "action_alignment": "next_state",
                    "state_semantics": "state_target",
                    "adapter": "policy/RDT/scripts/process_data.py",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _load_module(name: str, path: Path):
    if not path.is_file():
        raise AdapterError(f"找不到 policy 转换脚本: {path}")
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法创建模块 spec: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ImportError as exc:
        raise AdapterError(f"加载 policy 转换依赖失败: {path}: {exc}") from exc


ADAPTERS: dict[str, Callable[[], PolicyAdapter]] = {
    "act": ActAdapter,
    "dexvla": DexVlaAdapter,
    "dp": DpAdapter,
    "dp3": Dp3Adapter,
    "go1": Go1Adapter,
    "pi0": Pi0Adapter,
    "pi05": Pi05Adapter,
    "rdt": RdtAdapter,
    "tinyvla": TinyVlaAdapter,
}


def _build_parser(adapter: PolicyAdapter | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            adapter.description
            if adapter
            else "将 RoboTwin 原始数据转换为 Policy 训练数据"
        ),
    )
    parser.add_argument(
        "--policy",
        choices=sorted(ADAPTERS),
        required=True,
        help="目标 Policy",
    )
    if adapter is not None:
        adapter.add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--policy", choices=sorted(ADAPTERS))
    known, _ = bootstrap.parse_known_args(argv)
    if known.policy is None:
        _build_parser().parse_args(argv)
        return
    adapter = ADAPTERS[known.policy]()
    args = _build_parser(adapter).parse_args(argv)
    adapter.run(args)


if __name__ == "__main__":
    try:
        main()
    except ConversionError as exc:
        print(f"data_convert: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
