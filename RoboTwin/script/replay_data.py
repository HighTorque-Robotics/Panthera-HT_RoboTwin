"""Replay a collected RoboTwin trajectory in the SAPIEN third-person viewer.

The HDF5 file is used to identify and validate the episode.  The actual replay
uses RoboTwin's saved dense planner paths from ``_traj_data`` and the original
episode seed, which is the same source used by the normal collection replay
stage.  This avoids interpolating the down-sampled HDF5 drive targets.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py


ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from script.collect_data import prepare_task_and_args  # noqa: E402


EPISODE_PATTERN = re.compile(r"episode(?P<index>\d+)\.hdf5$")


class ReplayInputError(ValueError):
    """Raised when a raw RoboTwin episode cannot be replayed exactly."""


def resolve_episode_paths(hdf5_path: str | Path) -> dict[str, Path | int | str]:
    """Resolve the raw RoboTwin directory and episode index from an HDF5 path."""

    episode_path = Path(hdf5_path).expanduser().resolve()
    match = EPISODE_PATTERN.fullmatch(episode_path.name)
    if match is None:
        raise ReplayInputError(
            f"Expected an episode file named episodeN.hdf5, got {episode_path.name!r}"
        )
    if episode_path.parent.name != "data":
        raise ReplayInputError(
            f"Expected the HDF5 file under a data/ directory, got {episode_path.parent}"
        )

    raw_root = episode_path.parent.parent
    task_dir = raw_root.parent
    if not task_dir.name or not raw_root.name:
        raise ReplayInputError(f"Cannot infer task/config from {episode_path}")

    episode_index = int(match.group("index"))
    return {
        "episode_path": episode_path,
        "raw_root": raw_root,
        "task_name": task_dir.name,
        "task_config": raw_root.name,
        "episode_index": episode_index,
        "seed_path": raw_root / "seed.txt",
        "traj_path": raw_root / "_traj_data" / f"episode{episode_index}.pkl",
    }


def load_episode_seed(seed_path: str | Path, episode_index: int) -> int:
    """Return the original collection seed for an episode index."""

    seed_file = Path(seed_path)
    if not seed_file.is_file():
        raise ReplayInputError(f"Missing original seed list: {seed_file}")
    try:
        seeds = [int(token) for token in seed_file.read_text(encoding="utf-8").split()]
    except ValueError as exc:
        raise ReplayInputError(f"Invalid integer in seed list: {seed_file}") from exc
    if episode_index >= len(seeds):
        raise ReplayInputError(
            f"Seed list {seed_file} has {len(seeds)} entries, cannot replay episode {episode_index}"
        )
    return seeds[episode_index]


def _has_dataset(handle: h5py.File, path: str) -> bool:
    return path in handle and isinstance(handle[path], h5py.Dataset)


def validate_hdf5_schema(hdf5_path: str | Path, arm_mode: str) -> None:
    """Validate the raw HDF5 fields required by the configured arm mode."""

    with h5py.File(hdf5_path, "r") as handle:
        if arm_mode == "single":
            required = ("joint_action/arm", "joint_action/gripper")
            metadata_mode = handle.attrs.get("arm_mode")
            if isinstance(metadata_mode, bytes):
                metadata_mode = metadata_mode.decode("utf-8")
            if metadata_mode is not None and metadata_mode != "single":
                raise ReplayInputError(
                    f"HDF5 metadata arm_mode={metadata_mode!r} does not match single-arm config"
                )
        else:
            required = (
                "joint_action/left_arm",
                "joint_action/left_gripper",
                "joint_action/right_arm",
                "joint_action/right_gripper",
            )

        missing = [path for path in required if not _has_dataset(handle, path)]
        if missing:
            raise ReplayInputError(
                f"HDF5 {hdf5_path} is missing {arm_mode}-arm fields: {', '.join(missing)}"
            )


def validate_dense_trajectory(traj_data: dict, arm_mode: str, traj_path: Path) -> None:
    """Validate the dense native planner path saved during collection."""

    if arm_mode == "single":
        required = ("arm_joint_path",)
    else:
        required = ("left_joint_path", "right_joint_path")

    missing = [key for key in required if key not in traj_data]
    if missing:
        raise ReplayInputError(
            f"Native trajectory {traj_path} is missing {arm_mode}-arm fields: {', '.join(missing)}"
        )
    if any(not isinstance(traj_data[key], list) or not traj_data[key] for key in required):
        raise ReplayInputError(f"Native trajectory {traj_path} has empty planner paths")


def _configure_viewer_args(
    args: dict,
    raw_root: Path,
    render_freq: int,
    camera_xyz: list[float] | None,
    camera_rpy: list[float] | None,
) -> None:
    """Set replay-only options without changing the collection configuration."""

    args["save_path"] = str(raw_root)
    args["need_plan"] = False
    args["save_data"] = False
    args["collect_data"] = False
    args["render_freq"] = render_freq
    if camera_xyz is not None:
        args["camera_xyz_x"], args["camera_xyz_y"], args["camera_xyz_z"] = camera_xyz
    if camera_rpy is not None:
        args["camera_rpy_r"], args["camera_rpy_p"], args["camera_rpy_y"] = camera_rpy


def replay_episode(
    hdf5_path: str | Path,
    *,
    render_freq: int = 5,
    camera_xyz: list[float] | None = None,
    camera_rpy: list[float] | None = None,
    hold_viewer: bool = True,
) -> None:
    """Replay one raw RoboTwin episode using the native dense trajectory."""

    if render_freq <= 0:
        raise ReplayInputError("render_freq must be positive for Viewer replay")

    paths = resolve_episode_paths(hdf5_path)
    raw_root = paths["raw_root"]
    task_name = paths["task_name"]
    task_config = paths["task_config"]
    episode_index = paths["episode_index"]

    if not paths["episode_path"].is_file():
        raise ReplayInputError(f"HDF5 episode does not exist: {paths['episode_path']}")
    if not paths["traj_path"].is_file():
        raise ReplayInputError(
            "Exact replay requires the native dense trajectory next to the HDF5: "
            f"{paths['traj_path']}"
        )
    seed = load_episode_seed(paths["seed_path"], episode_index)

    task, args = prepare_task_and_args(task_name, task_config)
    _configure_viewer_args(args, raw_root, render_freq, camera_xyz, camera_rpy)
    validate_hdf5_schema(paths["episode_path"], args["arm_mode"])

    viewer = None
    try:
        task.setup_demo(now_ep_num=episode_index, seed=seed, **args)
        viewer = getattr(task, "viewer", None)
        traj_data = task.load_tran_data(episode_index)
        validate_dense_trajectory(traj_data, args["arm_mode"], paths["traj_path"])

        if args["arm_mode"] == "single":
            args["arm_joint_path"] = traj_data["arm_joint_path"]
        else:
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
        task.set_path_lst(args)

        print(
            f"Replay: task={task_name}, config={task_config}, "
            f"episode={episode_index}, seed={seed}, arm_mode={args['arm_mode']}"
        )
        task.play_once()
        print(f"Replay finished: plan_success={task.plan_success}, success={task.check_success()}")

        if hold_viewer and viewer is not None:
            print("Viewer replay finished. Close the SAPIEN window to exit.")
            while not viewer.closed:
                task._update_render()
                viewer.render()
    finally:
        if viewer is not None and not viewer.closed:
            viewer.close()
        if hasattr(task, "scene"):
            task.close_env(clear_cache=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a raw RoboTwin episode with the native dense trajectory in SAPIEN Viewer."
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to data/episodeN.hdf5")
    parser.add_argument(
        "--render-freq",
        type=int,
        default=5,
        help="Viewer refresh interval in simulation steps (default: 5)",
    )
    parser.add_argument(
        "--camera-xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Override the third-person Viewer position",
    )
    parser.add_argument(
        "--camera-rpy",
        nargs=3,
        type=float,
        metavar=("R", "P", "Y"),
        help="Override the third-person Viewer roll/pitch/yaw",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="Close immediately after the replay instead of waiting for the Viewer window",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parsed = parser.parse_args()
    try:
        replay_episode(
            parsed.hdf5_path,
            render_freq=parsed.render_freq,
            camera_xyz=parsed.camera_xyz,
            camera_rpy=parsed.camera_rpy,
            hold_viewer=not parsed.no_hold,
        )
    except ReplayInputError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
