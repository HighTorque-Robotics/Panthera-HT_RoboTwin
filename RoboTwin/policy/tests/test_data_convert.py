import argparse
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy import data_convert


class DataConvertTest(unittest.TestCase):
    def test_expected_policy_adapters_are_registered(self):
        self.assertEqual(
            set(data_convert.ADAPTERS),
            {
                "act",
                "pi0",
                "pi05",
                "go1",
                "rdt",
                "tinyvla",
                "dexvla",
                "dp",
                "dp3",
            },
        )

    def test_pi_adapters_call_their_own_native_process_data(self):
        self.assertIn(
            "policy/pi0/scripts/process_data.py",
            str(data_convert.Pi0Adapter.process_data_path),
        )
        self.assertIn(
            "policy/pi05/scripts/process_data.py",
            str(data_convert.Pi05Adapter.process_data_path),
        )

    def test_selected_episode_and_instruction_are_renumbered_by_symlink(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            (source / "data").mkdir(parents=True)
            (source / "instructions").mkdir()
            with h5py.File(source / "data" / "episode7.hdf5", "w"):
                pass
            instruction = {"seen": ["episode seven"]}
            (source / "instructions" / "episode7.json").write_text(
                json.dumps(instruction),
                encoding="utf-8",
            )

            dataset = data_convert.RobotTwinDataset.load(source, [7])
            staged = root / "staged"
            data_convert._link_episodes(
                dataset,
                staged,
                nested_data=True,
                include_instructions=True,
            )

            self.assertTrue((staged / "data" / "episode0.hdf5").is_symlink())
            self.assertEqual(
                json.loads((staged / "instructions" / "episode0.json").read_text()),
                instruction,
            )

    def test_dp3_rejects_empty_pointcloud_before_native_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            (source / "data").mkdir(parents=True)
            with h5py.File(source / "data" / "episode0.hdf5", "w") as episode:
                episode.create_dataset("joint_action/vector", data=np.zeros((2, 14)))
                episode.create_dataset("pointcloud", data=np.zeros((2, 0)))
            dataset = data_convert.RobotTwinDataset.load(source)
            output = root / "output"
            output.mkdir()

            with self.assertRaisesRegex(data_convert.InputError, "/pointcloud 为空"):
                data_convert.Dp3Adapter().convert(
                    dataset,
                    output,
                    argparse.Namespace(),
                )

    def test_tinyvla_requires_task_prompt_in_native_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            (source / "data").mkdir(parents=True)
            with h5py.File(source / "data" / "episode0.hdf5", "w"):
                pass
            dataset = data_convert.RobotTwinDataset.load(source)
            output = root / "output"
            output.mkdir()

            with self.assertRaisesRegex(data_convert.InputError, "未定义任务提示词"):
                data_convert.TinyVlaAdapter().convert(
                    dataset,
                    output,
                    argparse.Namespace(task_name="not_defined"),
                )


if __name__ == "__main__":
    unittest.main()
