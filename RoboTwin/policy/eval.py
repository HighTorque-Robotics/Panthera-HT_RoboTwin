"""Unified policy evaluation entry point.

The public entry point dispatches to RoboTwin's policy-specific deployment
adapter while keeping the simulator rollout loop in ``script/eval_policy.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


class EvaluationError(RuntimeError):
    """Raised when an evaluation request cannot be executed."""


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_robotwin_python() -> Path:
    project_python = REPO_ROOT / ".robotwin-runtime" / "envs" / "RoboTwin" / "bin" / "python"
    if project_python.is_file():
        return project_python
    return Path(sys.executable)


class ActEvalAdapter:
    name = "act"

    def __init__(self, request: argparse.Namespace) -> None:
        self.request = request
        self.checkpoint_dir = request.checkpoint_dir.expanduser().resolve()
        self.python = (
            request.python.expanduser().resolve()
            if request.python
            else _default_robotwin_python()
        )
        self.eval_script = REPO_ROOT / "script" / "eval_policy.py"
        self.deploy_config = REPO_ROOT / "policy" / "ACT" / "deploy_policy.yml"
        self.task_module = REPO_ROOT / "envs" / f"{request.task}.py"
        self.task_config = REPO_ROOT / "task_config" / f"{request.task_config}.yml"

    def validate(self) -> None:
        if self.request.episodes <= 0:
            raise EvaluationError("--episodes 必须为正数")
        if self.request.task != Path(self.request.task).name:
            raise EvaluationError("--task 必须是 envs 下的任务模块名，不能包含路径")
        if self.request.task_config != Path(self.request.task_config).name:
            raise EvaluationError("--task-config 必须是 task_config 下的配置名，不能包含路径")
        if not self.python.is_file():
            raise EvaluationError(f"Python 解释器不存在: {self.python}")
        if not self.checkpoint_dir.is_dir():
            raise EvaluationError(f"checkpoint 目录不存在: {self.checkpoint_dir}")
        checkpoint = self.checkpoint_dir / self.request.checkpoint_name
        if not checkpoint.is_file():
            raise EvaluationError(f"ACT checkpoint 不存在: {checkpoint}")
        stats = self.checkpoint_dir / "dataset_stats.pkl"
        if not stats.is_file():
            raise EvaluationError(f"ACT 评测缺少归一化统计量: {stats}")
        if not self.task_module.is_file():
            raise EvaluationError(f"RoboTwin 任务不存在: {self.task_module}")
        if not self.task_config.is_file():
            raise EvaluationError(f"RoboTwin 任务配置不存在: {self.task_config}")
        if not self.eval_script.is_file():
            raise EvaluationError(f"RoboTwin 评测脚本不存在: {self.eval_script}")
        if not self.deploy_config.is_file():
            raise EvaluationError(f"ACT 部署配置不存在: {self.deploy_config}")

    def command(self) -> list[str]:
        command = [
            str(self.python),
            str(self.eval_script.relative_to(REPO_ROOT)),
            "--config",
            str(self.deploy_config.relative_to(REPO_ROOT)),
            "--overrides",
            "--task_name",
            self.request.task,
            "--task_config",
            self.request.task_config,
            "--ckpt_setting",
            self.checkpoint_dir.name,
            "--ckpt_dir",
            str(self.checkpoint_dir),
            "--checkpoint_name",
            self.request.checkpoint_name,
            "--seed",
            str(self.request.seed),
            "--test_num",
            str(self.request.episodes),
            "--render_freq",
            "1" if self.request.render else "0",
            "--eval_video_log",
            "False",
            "--temporal_agg",
            str(self.request.temporal_agg),
        ]
        if self.request.fast_preview:
            command.extend(
                [
                    "--render_spp",
                    "1",
                    "--render_path_depth",
                    "2",
                    "--render_denoiser",
                    "none",
                ]
            )
        return command

    def run(self) -> None:
        self.validate()
        command = self.command()
        print(
            f"[policy.eval] CUDA_VISIBLE_DEVICES={self.request.gpu_id} "
            f"{shlex.join(command)}",
            flush=True,
        )
        if self.request.dry_run:
            return

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = self.request.gpu_id
        environment.setdefault(
            "TORCH_HOME",
            str(REPO_ROOT / ".robotwin-runtime" / "torch-cache"),
        )
        environment.setdefault(
            "MPLCONFIGDIR",
            str(REPO_ROOT / ".robotwin-runtime" / "matplotlib-cache"),
        )
        try:
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
        except subprocess.CalledProcessError as exc:
            raise EvaluationError(f"ACT 评测失败，退出码: {exc.returncode}") from exc


ADAPTERS = {"act": ActEvalAdapter}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboTwin policy 评测入口")
    parser.add_argument("--policy", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--task", required=True, help="RoboTwin 任务模块名")
    parser.add_argument("--task-config", required=True, help="task_config 下的配置名")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="包含 ACT checkpoint 和 dataset_stats.pkl 的训练输出目录",
    )
    parser.add_argument("--checkpoint-name", default="policy_best.ckpt")
    parser.add_argument("--episodes", type=int, default=1, help="闭环评测 episode 数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--python", type=Path, help="policy 环境的 Python；默认使用 RoboTwin 环境")
    parser.add_argument("--render", action="store_true", help="打开 SAPIEN 实时 viewer")
    parser.add_argument(
        "--fast-preview",
        action="store_true",
        help="降低光追质量并关闭去噪，仅用于快速链路检查",
    )
    parser.add_argument(
        "--temporal-agg",
        action="store_true",
        help="启用 ACT 时间聚合；默认关闭以减少推理次数",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印底层命令，不执行")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        ADAPTERS[args.policy](args).run()
    except EvaluationError as exc:
        raise SystemExit(f"评测入口错误: {exc}") from None


if __name__ == "__main__":
    main()
