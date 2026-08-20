"""Unified training entry point.

The public entry point dispatches policy-specific training commands. It does
not reimplement model training or data preprocessing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


class TrainingError(RuntimeError):
    """Raised when a training request cannot be executed."""


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_openpi_python() -> Path:
    project_python = REPO_ROOT / ".robotwin-runtime" / "envs" / "OpenPI" / "bin" / "python"
    if project_python.is_file():
        return project_python
    return Path(sys.executable)


def _default_act_python() -> Path:
    project_python = REPO_ROOT / ".robotwin-runtime" / "envs" / "RoboTwin" / "bin" / "python"
    if project_python.is_file():
        return project_python
    return Path(sys.executable)


class Pi05TrainAdapter:
    name = "pi05"
    default_config = "pi05_panthera_full_base"

    def __init__(self, request: argparse.Namespace) -> None:
        self.request = request
        self.dataset = request.dataset.resolve()
        self.output = request.output.resolve()
        self.python = Path(request.python).expanduser().resolve() if request.python else _default_openpi_python()
        self.config = request.config or self.default_config
        self.repo_id = self.dataset.name
        self.policy_root = REPO_ROOT / "policy" / "pi05"
        self.assets_base_dir = self.output / "assets"
        self.checkpoint_base_dir = self.output / "checkpoints"
        self.exp_name = request.exp_name or self.dataset.name

    def validate(self) -> None:
        if self.request.stage not in ("stats", "train"):
            raise TrainingError("Pi0.5 必须显式指定 --stage stats 或 --stage train")
        if self.request.epochs is not None:
            raise TrainingError("--epochs 仅适用于 ACT；Pi0.5 请使用 --steps")
        if not self.dataset.is_dir():
            raise TrainingError(f"数据集目录不存在: {self.dataset}")
        if not (self.dataset / "meta" / "info.json").is_file():
            raise TrainingError(f"不是有效的 LeRobot 数据集，缺少 meta/info.json: {self.dataset}")
        if not self.python.is_file():
            raise TrainingError(f"Python 解释器不存在: {self.python}")
        if not self.policy_root.is_dir():
            raise TrainingError(f"Pi0.5 源码目录不存在: {self.policy_root}")
        if self.request.batch_size is not None and self.request.batch_size <= 0:
            raise TrainingError("--batch-size 必须为正数")
        if self.request.num_workers < 0:
            raise TrainingError("--num-workers 不能为负数")
        if self.request.steps is not None and self.request.steps <= 0:
            raise TrainingError("--steps 必须为正数")
        if self.request.max_frames is not None and self.request.max_frames <= 0:
            raise TrainingError("--max-frames 必须为正数")
        if self.request.resume and self.request.overwrite:
            raise TrainingError("--resume 与 --overwrite 不能同时使用")
        if self.request.stage == "train" and not self._norm_stats_path().is_file():
            raise TrainingError(
                f"缺少 norm stats: {self._norm_stats_path()}，请先执行 --stage stats"
            )
        if self.output.exists() and not self.request.overwrite and not self.request.resume:
            if self.request.stage != "train" or not self._norm_stats_path().is_file():
                raise TrainingError(f"输出目录已存在，若要覆盖请添加 --overwrite: {self.output}")
            checkpoint_dir = self.checkpoint_base_dir / self.config / self.exp_name
            if checkpoint_dir.exists():
                raise TrainingError(
                    f"训练检查点已存在，请使用 --resume 继续或 --overwrite 覆盖: {checkpoint_dir}"
                )

    def run(self) -> None:
        self.validate()
        if not self.request.dry_run:
            self.output.mkdir(parents=True, exist_ok=True)

        if self.request.stage == "stats":
            command = self._stats_command()
        else:
            command = self._train_command()

        print(f"[policy.train] {shlex.join(command)}", flush=True)
        if not self.request.dry_run:
            environment = self._environment()
            try:
                subprocess.run(command, cwd=self.policy_root, env=environment, check=True)
            except subprocess.CalledProcessError as exc:
                raise TrainingError(f"Pi0.5 {self.request.stage} 失败，退出码: {exc.returncode}") from exc
            self._write_manifest(command)
            print(f"训练阶段完成: stage={self.request.stage}, output={self.output}", flush=True)

    def _stats_command(self) -> list[str]:
        batch_size = self.request.batch_size
        if batch_size is None and self.request.max_frames is not None:
            batch_size = min(64, self.request.max_frames)
        command = [
            str(self.python),
            "scripts/compute_norm_stats.py",
            "--config-name",
            self.config,
            "--repo-id",
            self.repo_id,
            "--assets-base-dir",
            str(self.assets_base_dir),
            "--num-workers",
            str(self.request.num_workers),
        ]
        if batch_size is not None:
            command.extend(["--batch-size", str(batch_size)])
        if self.request.max_frames is not None:
            command.extend(["--max-frames", str(self.request.max_frames)])
        return command

    def _train_command(self) -> list[str]:
        batch_size = self.request.batch_size or 64
        steps = self.request.steps or 20_000
        command = [
            str(self.python),
            "scripts/train.py",
            self.config,
            "--data.repo-id",
            self.repo_id,
            "--assets-base-dir",
            str(self.assets_base_dir),
            "--checkpoint-base-dir",
            str(self.checkpoint_base_dir),
            "--exp-name",
            self.exp_name,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(self.request.num_workers),
            "--num-train-steps",
            str(steps),
            "--wandb-enabled" if self.request.wandb else "--no-wandb-enabled",
        ]
        if self.request.overwrite:
            command.append("--overwrite")
        if self.request.resume:
            command.append("--resume")
        return command

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["HF_LEROBOT_HOME"] = str(self.dataset.parent)
        environment.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / ".robotwin-runtime" / "openpi-cache"))
        environment.setdefault("HF_HOME", str(REPO_ROOT / ".robotwin-runtime" / "hf-cache"))
        environment.setdefault(
            "HF_DATASETS_CACHE",
            str(REPO_ROOT / ".robotwin-runtime" / "hf-cache" / "datasets"),
        )
        source_paths = [
            str(self.policy_root / "src"),
            str(self.policy_root / "packages" / "openpi-client" / "src"),
        ]
        if environment.get("PYTHONPATH"):
            source_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(source_paths)
        return environment

    def _norm_stats_path(self) -> Path:
        return self.assets_base_dir / self.config / self.repo_id / "norm_stats.json"

    def _write_manifest(self, command: list[str]) -> None:
        manifest: dict[str, Any] = {
            "policy": self.name,
            "stage": self.request.stage,
            "dataset": str(self.dataset),
            "repo_id": self.repo_id,
            "config": self.config,
            "python": str(self.python),
            "command": shlex.join(command),
            "assets_base_dir": str(self.assets_base_dir),
            "checkpoint_base_dir": str(self.checkpoint_base_dir),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "train_run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class ActTrainAdapter:
    name = "act"
    camera_names = ("cam_high", "cam_right_wrist", "cam_left_wrist")
    kl_weight = 10
    chunk_size = 50
    hidden_dim = 512
    dim_feedforward = 3200
    learning_rate = 1e-5
    state_dim = 14
    save_freq = 2000

    def __init__(self, request: argparse.Namespace) -> None:
        self.request = request
        self.dataset = request.dataset.resolve()
        self.output = request.output.resolve()
        self.python = Path(request.python).expanduser().resolve() if request.python else _default_act_python()
        self.policy_root = REPO_ROOT / "policy" / "ACT"
        self.batch_size = request.batch_size if request.batch_size is not None else 1
        self.num_episodes = 0
        self.episode_len = 0
        self.task_name = ""

    def validate(self) -> None:
        if self.request.stage is not None:
            raise TrainingError("ACT 不使用 --stage；启动训练时会在内部计算统计量")
        if self.request.config is not None:
            raise TrainingError("--config 当前仅适用于 Pi0.5")
        if self.request.steps is not None:
            raise TrainingError("--steps 仅适用于 Pi0.5；ACT 请使用 --epochs")
        if self.request.epochs is None or self.request.epochs <= 0:
            raise TrainingError("ACT 必须显式指定正数 --epochs")
        if self.batch_size <= 0:
            raise TrainingError("--batch-size 必须为正数")
        if self.request.max_frames is not None:
            raise TrainingError("--max-frames 当前仅适用于 Pi0.5 stats 阶段")
        if self.request.resume:
            raise TrainingError("ACT 原训练脚本不支持 --resume")
        if self.request.overwrite:
            raise TrainingError("ACT 第一版不支持 --overwrite；请使用新的输出目录")
        if self.request.wandb:
            raise TrainingError("ACT 原训练脚本不支持 Weights & Biases")
        if self.request.num_workers != 0:
            raise TrainingError("ACT 当前由原 DataLoader 管理 worker，不支持 --num-workers")
        if not self.dataset.is_dir():
            raise TrainingError(f"数据集目录不存在: {self.dataset}")
        if not self.python.is_file():
            raise TrainingError(f"Python 解释器不存在: {self.python}")
        if not self.policy_root.is_dir():
            raise TrainingError(f"ACT 源码目录不存在: {self.policy_root}")
        if self.output.exists():
            raise TrainingError(f"ACT 输出目录已存在，请使用新的目录: {self.output}")
        self._load_dataset_contract()
        self._resolve_task_config()

    def _load_dataset_contract(self) -> None:
        manifest_path = self.dataset / "robotwin_conversion.json"
        if not manifest_path.is_file():
            raise TrainingError(f"ACT 数据集缺少转换清单: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingError(f"无法读取 ACT 转换清单 {manifest_path}: {exc}") from exc

        if manifest.get("policy") != "act":
            raise TrainingError(f"转换清单不是 ACT 数据集: {manifest_path}")
        if manifest.get("action_dim") != 14:
            raise TrainingError(f"ACT 双臂数据 action_dim 必须为 14: {manifest_path}")
        if tuple(manifest.get("camera_names", ())) != self.camera_names:
            raise TrainingError(
                "ACT 相机顺序必须为 " + ", ".join(self.camera_names)
            )

        episodes = manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise TrainingError(f"ACT 转换清单没有 episode: {manifest_path}")
        try:
            output_ids = sorted(int(episode["output_episode_id"]) for episode in episodes)
            output_lengths = [int(episode["output_num_frames"]) for episode in episodes]
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingError(f"ACT 转换清单 episode 字段无效: {manifest_path}") from exc
        expected_ids = list(range(len(episodes)))
        if output_ids != expected_ids:
            raise TrainingError("ACT output_episode_id 必须从 0 开始连续编号")
        if any(length <= 0 for length in output_lengths):
            raise TrainingError("ACT episode 的 output_num_frames 必须为正数")
        for episode_id in expected_ids:
            episode_path = self.dataset / f"episode_{episode_id}.hdf5"
            if not episode_path.is_file():
                raise TrainingError(f"ACT 数据集缺少 episode 文件: {episode_path}")

        self.num_episodes = len(episodes)
        self.episode_len = max(output_lengths)

    def _resolve_task_config(self) -> None:
        registry_path = self.policy_root / "SIM_TASK_CONFIGS.json"
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingError(f"无法读取 ACT 数据集注册表 {registry_path}: {exc}") from exc
        if not isinstance(registry, dict):
            raise TrainingError(f"ACT 数据集注册表必须是 JSON 对象: {registry_path}")

        matches = []
        for task_name, task_config in registry.items():
            if not isinstance(task_config, dict) or "dataset_dir" not in task_config:
                continue
            registered_dataset = Path(task_config["dataset_dir"])
            if not registered_dataset.is_absolute():
                registered_dataset = (self.policy_root / registered_dataset).resolve()
            if registered_dataset == self.dataset:
                matches.append((task_name, task_config))

        if not matches:
            raise TrainingError(
                f"ACT 数据集未注册到 {registry_path}: {self.dataset}"
            )
        if len(matches) > 1:
            names = ", ".join(task_name for task_name, _ in matches)
            raise TrainingError(f"ACT 数据集匹配到多个 task 配置: {names}")

        task_name, task_config = matches[0]
        if not task_name.startswith("sim-"):
            raise TrainingError(f"ACT 仿真 task_name 必须以 sim- 开头: {task_name}")
        if task_config.get("num_episodes") != self.num_episodes:
            raise TrainingError(
                f"ACT 注册的 num_episodes 与转换清单不一致: "
                f"{task_config.get('num_episodes')} != {self.num_episodes}"
            )
        if task_config.get("episode_len") != self.episode_len:
            raise TrainingError(
                f"ACT 注册的 episode_len 与转换清单不一致: "
                f"{task_config.get('episode_len')} != {self.episode_len}"
            )
        if tuple(task_config.get("camera_names", ())) != self.camera_names:
            raise TrainingError("ACT 注册的 camera_names 与转换清单不一致")
        self.task_name = task_name

    def run(self) -> None:
        self.validate()
        command = self._train_command()
        print(f"[policy.train] {shlex.join(command)}", flush=True)
        if self.request.dry_run:
            return

        try:
            subprocess.run(
                command,
                cwd=self.policy_root,
                env=self._environment(),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise TrainingError(f"ACT 训练失败，退出码: {exc.returncode}") from exc
        self._write_manifest(command)
        print(f"训练完成: policy=act, output={self.output}", flush=True)

    def _train_command(self) -> list[str]:
        return [
            str(self.python),
            "imitate_episodes.py",
            "--task_name",
            self.task_name,
            "--ckpt_dir",
            str(self.output),
            "--policy_class",
            "ACT",
            "--kl_weight",
            str(self.kl_weight),
            "--chunk_size",
            str(self.chunk_size),
            "--hidden_dim",
            str(self.hidden_dim),
            "--batch_size",
            str(self.batch_size),
            "--dim_feedforward",
            str(self.dim_feedforward),
            "--num_epochs",
            str(self.request.epochs),
            "--lr",
            str(self.learning_rate),
            "--save_freq",
            str(self.save_freq),
            "--state_dim",
            str(self.state_dim),
            "--seed",
            str(self.request.seed),
        ]

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.setdefault(
            "TORCH_HOME",
            str(REPO_ROOT / ".robotwin-runtime" / "torch-cache"),
        )
        environment.setdefault(
            "MPLCONFIGDIR",
            str(REPO_ROOT / ".robotwin-runtime" / "matplotlib-cache"),
        )
        return environment

    def _write_manifest(self, command: list[str]) -> None:
        manifest: dict[str, Any] = {
            "policy": self.name,
            "dataset": str(self.dataset),
            "output": str(self.output),
            "python": str(self.python),
            "command": shlex.join(command),
            "task_name": self.task_name,
            "num_episodes": self.num_episodes,
            "episode_len": self.episode_len,
            "camera_names": list(self.camera_names),
            "epochs": self.request.epochs,
            "batch_size": self.batch_size,
            "seed": self.request.seed,
            "model": {
                "state_dim": self.state_dim,
                "kl_weight": self.kl_weight,
                "chunk_size": self.chunk_size,
                "hidden_dim": self.hidden_dim,
                "dim_feedforward": self.dim_feedforward,
                "learning_rate": self.learning_rate,
            },
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "train_run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


ADAPTERS = {"act": ActTrainAdapter, "pi05": Pi05TrainAdapter}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboTwin policy 训练入口")
    parser.add_argument("--policy", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="对应 policy 的已转换数据集目录")
    parser.add_argument("--output", type=Path, required=True, help="训练产物目录")
    parser.add_argument("--stage", choices=("stats", "train"), help="Pi0.5 阶段；ACT 不使用")
    parser.add_argument("--config", help="Pi0.5 配置名")
    parser.add_argument("--python", type=Path, help="policy 环境的 Python；默认使用对应项目内环境")
    parser.add_argument("--exp-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, help="训练步数，仅 Pi0.5 train 阶段使用")
    parser.add_argument("--epochs", type=int, help="训练轮数，仅 ACT 使用")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，仅 ACT 使用")
    parser.add_argument("--max-frames", type=int, help="统计阶段最多使用的帧数")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="训练阶段启用 Weights & Biases")
    parser.add_argument("--dry-run", action="store_true", help="只打印底层命令，不执行")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        ADAPTERS[args.policy](args).run()
    except TrainingError as exc:
        raise SystemExit(f"训练入口错误: {exc}") from None


if __name__ == "__main__":
    main()
