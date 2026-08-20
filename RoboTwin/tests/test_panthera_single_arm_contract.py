import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import yaml

# Contract tests do not instantiate planners. Stub the planner module before
# importing Base_Task so test discovery does not initialize CUDA through CuRobo.
planner_stub = types.ModuleType("envs.robot.planner")
planner_stub.CuroboPlanner = type("CuroboPlanner", (), {})
planner_stub.MplibPlanner = type("MplibPlanner", (), {})
sys.modules.setdefault("envs.robot.planner", planner_stub)

from envs._base_task import Base_Task
from envs.camera.camera import Camera
from envs.utils import pkl2hdf5
from envs.utils.images_to_video import get_ffmpeg_executable
from script.validation.rules.move_pillbottle_pad import _entity_id


ROOT = Path(__file__).resolve().parents[1]


class _FakeEntity:
    def set_pose(self, pose):
        self.pose = pose


class _FakeCamera:
    def __init__(self, name):
        self.name = name
        self.entity = _FakeEntity()


class _FakeScene:
    def __init__(self):
        self.camera_names = []

    def add_camera(self, *, name, **kwargs):
        self.camera_names.append(name)
        return _FakeCamera(name)


class _ObservationCameras:
    def update_picture(self):
        pass

    def get_config(self):
        return {"head_camera": {}, "wrist_camera": {}}

    def get_rgb(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return {
            "head_camera": {"rgb": frame},
            "wrist_camera": {"rgb": frame},
        }


class _ObservationRobot:
    def get_left_arm_jointState(self):
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

    def get_left_gripper_val(self):
        return 1.0


def _camera_kwargs(arm_mode):
    embodiment = yaml.safe_load(
        (ROOT / "assets/embodiments/panthera-6dof/config.yml").read_text()
    )
    return {
        "arm_mode": arm_mode,
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D405",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        },
        "left_embodiment_config": embodiment,
    }


class PantheraSingleArmContractTest(unittest.TestCase):
    def test_physics_rule_uses_stable_sapien_entity_id(self):
        class EntityWrapper:
            def get_global_id(self):
                return 42

        self.assertIsNot(EntityWrapper(), EntityWrapper())
        self.assertEqual(_entity_id(EntityWrapper()), _entity_id(EntityWrapper()))

    def test_video_encoder_falls_back_to_imageio_ffmpeg(self):
        with mock.patch("shutil.which", return_value=None):
            executable = get_ffmpeg_executable()
        self.assertTrue(Path(executable).is_file())

    def test_single_config_preserves_collection_randomization(self):
        single = yaml.safe_load(
            (ROOT / "task_config/move_pillbottle_pad_panthera_single.yml").read_text()
        )
        dual = yaml.safe_load(
            (ROOT / "task_config/move_pillbottle_pad_panthera.yml").read_text()
        )
        self.assertEqual(single["arm_mode"], "single")
        self.assertEqual(single["embodiment"], ["panthera-6dof"])
        self.assertEqual(single["domain_randomization"], dual["domain_randomization"])
        self.assertEqual(single["camera"], dual["camera"])
        self.assertEqual(single["data_type"], dual["data_type"])

    def test_single_camera_has_one_generic_wrist_camera(self):
        scene = _FakeScene()
        camera = Camera(**_camera_kwargs("single"))
        camera.load_camera(scene)
        self.assertEqual(scene.camera_names.count("wrist_camera"), 1)
        self.assertNotIn("left_camera", scene.camera_names)
        self.assertNotIn("right_camera", scene.camera_names)

    def test_dual_camera_names_are_unchanged(self):
        scene = _FakeScene()
        camera = Camera(**_camera_kwargs("dual"))
        camera.load_camera(scene)
        self.assertEqual(scene.camera_names.count("left_camera"), 1)
        self.assertEqual(scene.camera_names.count("right_camera"), 1)
        self.assertNotIn("wrist_camera", scene.camera_names)

    def test_single_observation_is_native_seven_dimensional(self):
        task = Base_Task.__new__(Base_Task)
        task.single_arm_mode = True
        task.data_type = {
            "rgb": True,
            "third_view": False,
            "mesh_segmentation": False,
            "actor_segmentation": False,
            "depth": False,
            "endpose": True,
            "qpos": True,
            "pointcloud": False,
        }
        task.cameras = _ObservationCameras()
        task.robot = _ObservationRobot()
        task._update_render = lambda: None
        task.get_arm_pose = lambda arm: [0.0] * 7

        observation = task.get_obs()

        self.assertEqual(set(observation["observation"]), {"head_camera", "wrist_camera"})
        self.assertEqual(set(observation["joint_action"]), {"arm", "gripper", "vector"})
        self.assertEqual(observation["joint_action"]["vector"].shape, (7,))
        self.assertEqual(set(observation["endpose"]), {"arm", "gripper"})

    def test_single_trajectory_has_no_second_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Base_Task.__new__(Base_Task)
            task.single_arm_mode = True
            task.save_dir = directory
            task.left_joint_path = [{"status": "Success"}]
            task.right_joint_path = [{"status": "must-not-be-saved"}]
            task.save_traj_data(0)

            with open(Path(directory) / "_traj_data/episode0.pkl", "rb") as file:
                trajectory = pickle.load(file)

        self.assertEqual(trajectory["arm_mode"], "single")
        self.assertEqual(set(trajectory), {"schema_version", "arm_mode", "arm_joint_path"})

    def test_hdf5_single_schema_and_metadata(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        sample = {
            "observation": {
                "head_camera": {"rgb": frame},
                "wrist_camera": {"rgb": frame},
            },
            "pointcloud": np.empty((0, 6)),
            "joint_action": {
                "arm": np.arange(6, dtype=float),
                "gripper": 1.0,
                "vector": np.arange(7, dtype=float),
            },
            "endpose": {"arm": np.arange(7, dtype=float), "gripper": 1.0},
        }
        metadata = {
            "schema_version": "panthera-single-v1",
            "arm_mode": "single",
            "robot_type": "panthera-6dof",
            "state_dim": 7,
        }

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pkl_path = directory / "0.pkl"
            with pkl_path.open("wb") as file:
                pickle.dump(sample, file)
            hdf5_path = directory / "episode0.hdf5"
            with mock.patch.object(pkl2hdf5, "images_to_video") as video_writer:
                pkl2hdf5.pkl_files_to_hdf5_and_video(
                    [str(pkl_path)],
                    str(hdf5_path),
                    str(directory / "episode0.mp4"),
                    metadata=metadata,
                )
            self.assertEqual(video_writer.call_count, 2)
            with h5py.File(hdf5_path, "r") as file:
                self.assertEqual(file.attrs["schema_version"], "panthera-single-v1")
                self.assertEqual(file.attrs["arm_mode"], "single")
                self.assertEqual(file.attrs["robot_type"], "panthera-6dof")
                self.assertEqual(file.attrs["state_dim"], 7)
                self.assertEqual(file["joint_action/vector"].shape, (1, 7))
                self.assertEqual(set(file["joint_action"]), {"arm", "gripper", "vector"})
                self.assertEqual(set(file["observation"]), {"head_camera", "wrist_camera"})


if __name__ == "__main__":
    unittest.main()
