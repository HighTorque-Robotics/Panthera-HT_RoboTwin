"""Write Pi0.5 intermediate HDF5 episodes as a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterator

import cv2
import h5py
import numpy as np


DUAL_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DUAL_MOTORS = (
    *(f"left_joint_{i}" for i in range(6)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(6)),
    "right_gripper",
)
SINGLE_CAMERAS = ("cam_high", "cam_wrist")
SINGLE_MOTORS = tuple(f"joint_{i}" for i in range(6)) + ("gripper",)


def _decode_image_value(values: h5py.Dataset, index: int) -> np.ndarray:
    encoded = values[index]
    image = cv2.imdecode(np.frombuffer(bytes(encoded), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{values.name}[{index}] JPEG 解码失败")
    return image


def _load_images(ep: h5py.File, camera: str) -> np.ndarray:
    values = ep[f"/observations/images/{camera}"]
    if values.ndim == 4:
        images = values[:]
    else:
        images = []
        for index, encoded in enumerate(values):
            image = _decode_image_value(values, index)
            # RoboTwin passes its RGB array directly to OpenCV when encoding;
            # imdecode therefore already restores the original numeric order.
            images.append(image)
        images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"{camera} 图像 shape 非法: {images.shape}")
    return images


def _image_shape(values: h5py.Dataset, camera: str) -> tuple[int, int, int]:
    if values.ndim == 4:
        shape = values.shape[1:]
    else:
        shape = _decode_image_value(values, 0).shape
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"{camera} 图像 shape 非法: {shape}")
    return tuple(shape)


def _load_episode(
    path: Path, *, load_images: bool = True
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str, str]:
    with h5py.File(path, "r") as ep:
        state = np.asarray(ep["/observations/qpos"][:], dtype=np.float32)
        action = np.asarray(ep["/action"][:], dtype=np.float32)
        image_names = tuple(ep["/observations/images"].keys())
        if set(image_names) == set(SINGLE_CAMERAS):
            cameras = SINGLE_CAMERAS
            arm_mode = "single"
        elif set(image_names) == set(DUAL_CAMERAS):
            cameras = DUAL_CAMERAS
            arm_mode = "dual"
        else:
            raise ValueError(f"{path}: 不支持的相机集合: {image_names}")
        if load_images:
            images = {camera: _load_images(ep, camera) for camera in cameras}
        else:
            images = {}
            for camera in cameras:
                values = ep[f"/observations/images/{camera}"]
                if values.shape[0] != state.shape[0] or _image_shape(values, camera)[-1] != 3:
                    raise ValueError(f"{path}: {camera} 图像帧数或 shape 非法")
    instruction_path = path.parent / "instructions.json"
    try:
        instruction_data = json.loads(instruction_path.read_text(encoding="utf-8"))
        instructions = instruction_data["instructions"]
        instruction = instructions[0]
    except (OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: 无法读取 instructions.json") from exc
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{path}: instructions.json 中没有非空指令")
    expected_dim = 7 if arm_mode == "single" else 14
    if state.ndim != 2 or action.ndim != 2 or state.shape[1] != expected_dim or action.shape[1] != expected_dim:
        raise ValueError(f"{path}: state/action 必须为 [N,{expected_dim}]，实际为 {state.shape}, {action.shape}")
    if state.shape[0] != action.shape[0] or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: state/action 帧数不一致或含 NaN/Inf")
    if images and len({state.shape[0], *(image.shape[0] for image in images.values())}) != 1:
        raise ValueError(f"{path}: state/action/图像帧数不一致")
    return state, action, images, instruction, arm_mode


def _features(images: dict[str, np.ndarray], arm_mode: str) -> dict:
    cameras = SINGLE_CAMERAS if arm_mode == "single" else DUAL_CAMERAS
    motors = SINGLE_MOTORS if arm_mode == "single" else DUAL_MOTORS
    height, width = images[cameras[0]].shape[1:3]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [list(motors)]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [list(motors)]},
    }
    for camera in cameras:
        height_i, width_i = images[camera].shape[1:3]
        if (height_i, width_i) != (height, width):
            raise ValueError("三路相机输出分辨率不一致")
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


class LeRobotDatasetWriter:
    """Append processed episodes to one LeRobot dataset."""

    def __init__(
        self,
        *,
        output_root: Path,
        repo_id: str,
        first_episode: Path,
        fps: float,
        robot_type: str = "panthera-6dof-dual",
    ) -> None:
        if fps <= 0:
            raise ValueError("fps 必须是正数")
        old_home = os.environ.get("HF_LEROBOT_HOME")
        os.environ["HF_LEROBOT_HOME"] = str(output_root)
        try:
            from lerobot.common.datasets import lerobot_dataset as lerobot_module
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            _restore_home(old_home)
            raise RuntimeError("缺少 LeRobot 依赖，请在 Pi0.5 环境中运行") from exc

        self._old_home = old_home
        self._lerobot_module = lerobot_module
        self.output = output_root / repo_id
        if self.output.exists():
            shutil.rmtree(self.output)
        state, action, images, _, arm_mode = _load_episode(first_episode)
        del state, action
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=self.output,
            fps=fps,
            robot_type=robot_type,
            features=_features(images, arm_mode),
            use_videos=False,
        )

    def add_episode(self, path: Path) -> None:
        state, action, _, instruction, arm_mode = _load_episode(path, load_images=False)
        expected_mode = "single" if len(self.dataset.features["action"]["shape"]) == 1 and self.dataset.features["action"]["shape"][0] == 7 else "dual"
        if arm_mode != expected_mode:
            raise ValueError(f"{path}: arm_mode={arm_mode} 与数据集 schema={expected_mode} 不一致")
        cameras = SINGLE_CAMERAS if arm_mode == "single" else DUAL_CAMERAS
        with h5py.File(path, "r") as ep:
            image_values = {camera: ep[f"/observations/images/{camera}"] for camera in cameras}
            for index in range(state.shape[0]):
                frame = {
                    "observation.state": state[index],
                    "action": action[index],
                    "task": instruction,
                }
                for camera in cameras:
                    values = image_values[camera]
                    frame[f"observation.images.{camera}"] = (
                        values[index]
                        if values.ndim == 4
                        else _decode_image_value(values, index)
                    )
                self.dataset.add_frame(frame)
        self._save_episode_without_dataset_concat()
        del state, action

    def _save_episode_without_dataset_concat(self) -> None:
        """Persist one episode without growing LeRobot's in-memory dataset.

        The pinned LeRobot commit's ``save_episode`` concatenates the new Arrow
        table into ``self.hf_dataset`` on every episode. That is unnecessary for
        a writer and makes memory usage grow with the dataset size.
        """

        datasets = self._lerobot_module.datasets
        validate_episode_buffer = self._lerobot_module.validate_episode_buffer
        embed_images = self._lerobot_module.embed_images
        compute_episode_stats = self._lerobot_module.compute_episode_stats
        get_episode_data_index = self._lerobot_module.get_episode_data_index
        check_timestamps_sync = self._lerobot_module.check_timestamps_sync

        episode_buffer = self.dataset.episode_buffer
        episode_index = episode_buffer["episode_index"]
        validate_episode_buffer(episode_buffer, self.dataset.meta.total_episodes, self.dataset.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_buffer["index"] = np.arange(
            self.dataset.meta.total_frames,
            self.dataset.meta.total_frames + episode_length,
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.dataset.meta.get_task_index(task) is None:
                self.dataset.meta.add_task(task)
        episode_buffer["task_index"] = np.array(
            [self.dataset.meta.get_task_index(task) for task in tasks]
        )

        for key, feature in self.dataset.features.items():
            if key in ["index", "episode_index", "task_index"] or feature["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self.dataset._wait_image_writer()
        episode_dict = {key: episode_buffer[key] for key in self.dataset.hf_features}
        episode_dataset = datasets.Dataset.from_dict(
            episode_dict,
            features=self.dataset.hf_features,
            split="train",
        )
        episode_dataset = embed_images(episode_dataset)
        episode_path = self.dataset.root / self.dataset.meta.get_data_file_path(episode_index)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        episode_dataset.to_parquet(episode_path)

        episode_stats = compute_episode_stats(episode_buffer, self.dataset.features)
        self.dataset.meta.save_episode(episode_index, episode_length, episode_tasks, episode_stats)

        ep_data_index = get_episode_data_index(self.dataset.meta.episodes, [episode_index])
        ep_data_index_np = {key: value.numpy() for key, value in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.dataset.fps,
            self.dataset.tolerance_s,
        )

        image_dir = self.dataset.root / "images"
        if image_dir.is_dir():
            shutil.rmtree(image_dir)
        self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        del episode_dataset, episode_buffer

    def close(self) -> None:
        _restore_home(self._old_home)


def _restore_home(old_home: str | None) -> None:
    if old_home is None:
        os.environ.pop("HF_LEROBOT_HOME", None)
    else:
        os.environ["HF_LEROBOT_HOME"] = old_home


def convert_processed_data(
    raw_dir: Path,
    output_root: Path,
    repo_id: str,
    *,
    fps: float,
    robot_type: str = "panthera-6dof-dual",
) -> Path:
    if fps <= 0:
        raise ValueError("fps 必须是正数")
    files = sorted(raw_dir.glob("episode_*/*.hdf5"))
    if not files:
        raise ValueError(f"未发现中间 HDF5: {raw_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    writer = LeRobotDatasetWriter(
        output_root=output_root,
        repo_id=repo_id,
        first_episode=files[0],
        fps=fps,
        robot_type=robot_type,
    )
    try:
        for path in files:
            writer.add_episode(path)
    finally:
        writer.close()
    return writer.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()
    convert_processed_data(args.raw_dir, args.output_root, args.repo_id, fps=args.fps)
