import tempfile
import unittest
from pathlib import Path

import h5py

from script.replay_data import (
    ReplayInputError,
    load_episode_seed,
    replay_episode,
    resolve_episode_paths,
    validate_dense_trajectory,
    validate_hdf5_schema,
)


class ReplayDataTest(unittest.TestCase):
    def test_resolve_episode_paths_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "task_name" / "task_config"
            episode_path = root / "data" / "episode3.hdf5"
            episode_path.parent.mkdir(parents=True)
            episode_path.touch()
            seed_path = root / "seed.txt"
            seed_path.write_text("4 8 15 16\n", encoding="utf-8")

            paths = resolve_episode_paths(episode_path)
            self.assertEqual(paths["task_name"], "task_name")
            self.assertEqual(paths["task_config"], "task_config")
            self.assertEqual(paths["episode_index"], 3)
            self.assertEqual(load_episode_seed(seed_path, 3), 16)

    def test_hdf5_schema_for_single_and_dual(self):
        with tempfile.TemporaryDirectory() as directory:
            single_path = Path(directory) / "single.hdf5"
            with h5py.File(single_path, "w") as handle:
                handle.attrs["arm_mode"] = "single"
                handle.create_dataset("joint_action/arm", shape=(2, 6), dtype="f4")
                handle.create_dataset("joint_action/gripper", shape=(2,), dtype="f4")
            validate_hdf5_schema(single_path, "single")

            dual_path = Path(directory) / "dual.hdf5"
            with h5py.File(dual_path, "w") as handle:
                for name, shape in (
                    ("joint_action/left_arm", (2, 6)),
                    ("joint_action/left_gripper", (2,)),
                    ("joint_action/right_arm", (2, 6)),
                    ("joint_action/right_gripper", (2,)),
                ):
                    handle.create_dataset(name, shape=shape, dtype="f4")
            validate_hdf5_schema(dual_path, "dual")

    def test_schema_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode0.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["arm_mode"] = "dual"
                handle.create_dataset("joint_action/arm", shape=(2, 6), dtype="f4")
                handle.create_dataset("joint_action/gripper", shape=(2,), dtype="f4")
            with self.assertRaises(ReplayInputError):
                validate_hdf5_schema(path, "single")

    def test_missing_dense_trajectory_is_rejected_before_environment_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "task_name" / "task_config"
            episode_path = root / "data" / "episode0.hdf5"
            episode_path.parent.mkdir(parents=True)
            with h5py.File(episode_path, "w") as handle:
                handle.create_dataset("joint_action/arm", shape=(2, 6), dtype="f4")
                handle.create_dataset("joint_action/gripper", shape=(2,), dtype="f4")
            (root / "seed.txt").write_text("2\n", encoding="utf-8")

            with self.assertRaisesRegex(ReplayInputError, "Exact replay requires"):
                replay_episode(episode_path, hold_viewer=False)

    def test_dense_trajectory_schema(self):
        validate_dense_trajectory({"arm_joint_path": [{"status": "Success"}]}, "single", Path("traj.pkl"))
        validate_dense_trajectory(
            {
                "left_joint_path": [{"status": "Success"}],
                "right_joint_path": [{"status": "Success"}],
            },
            "dual",
            Path("traj.pkl"),
        )
        with self.assertRaises(ReplayInputError):
            validate_dense_trajectory({}, "single", Path("traj.pkl"))


if __name__ == "__main__":
    unittest.main()
