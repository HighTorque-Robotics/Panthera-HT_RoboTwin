import sys
import types
import unittest
from pathlib import Path

import numpy as np
import yaml

# These tests exercise task semantics without constructing CUDA planners.
planner_stub = types.ModuleType("envs.robot.planner")
planner_stub.CuroboPlanner = type("CuroboPlanner", (), {})
planner_stub.MplibPlanner = type("MplibPlanner", (), {})
sys.modules.setdefault("envs.robot.planner", planner_stub)

from envs.blocks_ranking_rgb import blocks_ranking_rgb
from envs.utils.action import ArmTag
from script.validation.rules import create_rule
from script.validation.rules.blocks_ranking_rgb import BlocksRankingRGBRule


ROOT = Path(__file__).resolve().parents[1]


class _Pose:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=float)


class _Block:
    def __init__(self, position):
        self.pose = _Pose(position)

    def get_pose(self):
        return self.pose


class _MatrixPose:
    def __init__(self, position):
        self.matrix = np.eye(4)
        self.matrix[:3, 3] = position

    def to_transformation_matrix(self):
        return self.matrix


class _PhysicsBlock:
    def __init__(self, position):
        self.pose = _MatrixPose(position)

    def get_pose(self):
        return self.pose


class _StationaryBody:
    def get_linear_velocity(self):
        return np.zeros(3)


def _task_for_pick(*, single_arm_mode, previous_arm=None):
    task = blocks_ranking_rgb.__new__(blocks_ranking_rgb)
    task.single_arm_mode = single_arm_mode
    task.active_block_index = None
    task.last_gripper = previous_arm
    task.moves = []
    task.grasp_actor = lambda block, arm_tag, **kwargs: (arm_tag, ["grasp"])
    task.move_by_displacement = lambda arm_tag, **kwargs: (arm_tag, ["displace"])
    task.place_actor = lambda block, arm_tag, **kwargs: (arm_tag, ["place"])
    task.move = lambda *actions: task.moves.append(actions) or True
    return task


class PantheraSingleArmBlocksRankingContractTest(unittest.TestCase):
    def test_physics_rule_is_registered(self):
        self.assertIsInstance(create_rule("blocks_ranking_rgb"), BlocksRankingRGBRule)

    def test_single_config_preserves_dual_randomization_and_sensors(self):
        single = yaml.safe_load(
            (ROOT / "task_config/blocks_ranking_rgb_panthera_single.yml").read_text()
        )
        dual = yaml.safe_load(
            (ROOT / "task_config/blocks_ranking_rgb_panthera.yml").read_text()
        )

        self.assertEqual(single["arm_mode"], "single")
        self.assertEqual(single["embodiment"], ["panthera-6dof"])
        self.assertTrue(single["physics_validation"])
        self.assertEqual(single["domain_randomization"], dual["domain_randomization"])
        self.assertEqual(single["camera"], dual["camera"])
        self.assertEqual(single["data_type"], dual["data_type"])

    def test_single_positive_x_block_uses_only_internal_left_channel(self):
        task = _task_for_pick(single_arm_mode=True, previous_arm=ArmTag("right"))
        task.back_to_origin = lambda arm_tag: self.fail(
            "single-arm mode must not schedule the dual-arm return-to-origin action"
        )

        language_arm = task.pick_and_place_block(
            _Block([0.25, 0.0, 0.765]),
            target_pose=[0.08, -0.15, 0.74, 0, 1, 0, 0],
            block_index=2,
        )

        self.assertEqual(language_arm, "robot")
        self.assertEqual(task.active_block_index, 2)
        self.assertEqual(task.last_gripper, ArmTag("left"))
        for move_call in task.moves:
            for arm_actions in move_call:
                self.assertEqual(arm_actions[0], ArmTag("left"))

    def test_dual_mode_keeps_position_based_arm_switch(self):
        task = _task_for_pick(single_arm_mode=False, previous_arm=ArmTag("left"))
        returned = []

        def back_to_origin(arm_tag):
            returned.append(arm_tag)
            return arm_tag, ["return"]

        task.back_to_origin = back_to_origin
        language_arm = task.pick_and_place_block(
            _Block([0.25, 0.0, 0.765]),
            target_pose=[0.08, -0.15, 0.74, 0, 1, 0, 0],
            block_index=2,
        )

        self.assertEqual(language_arm, "right")
        self.assertEqual(returned, [ArmTag("left")])
        self.assertEqual(task.last_gripper, ArmTag("right"))

    def test_single_success_does_not_read_second_gripper(self):
        task = blocks_ranking_rgb.__new__(blocks_ranking_rgb)
        task.single_arm_mode = True
        task.block1 = _Block([-0.08, -0.15, 0.765])
        task.block2 = _Block([0.00, -0.15, 0.765])
        task.block3 = _Block([0.08, -0.15, 0.765])
        task.is_left_gripper_open = lambda: True
        task.is_right_gripper_open = lambda: self.fail(
            "single-arm success must not read a second gripper"
        )

        self.assertTrue(task.check_success())

    def test_dual_success_still_requires_both_grippers(self):
        task = blocks_ranking_rgb.__new__(blocks_ranking_rgb)
        task.single_arm_mode = False
        task.block1 = _Block([-0.08, -0.15, 0.765])
        task.block2 = _Block([0.00, -0.15, 0.765])
        task.block3 = _Block([0.08, -0.15, 0.765])
        task.is_left_gripper_open = lambda: True
        task.is_right_gripper_open = lambda: False

        self.assertFalse(task.check_success())

    def test_physics_report_requires_every_block_cycle(self):
        rule = BlocksRankingRGBRule()
        rule.dt = 0.004
        rule.single_arm_mode = True
        rule.minimum_robot_separation = -0.0002
        rule.maximum_robot_contact_impulse = 0.1
        rule.maximum_robot_penetration = 0.0002
        rule.unexpected_robot_contacts = {}
        rule.self_contacts = {}
        rule.contact_pairs = {}
        rule.trace = []
        rule.blocks = []
        for index, label in enumerate(rule.LABELS):
            position = np.array([-0.08 + 0.08 * index, -0.15, 0.765])
            initial_pose = np.eye(4)
            initial_pose[:3, 3] = position
            rule.blocks.append(
                {
                    "label": label,
                    "actor": _PhysicsBlock(position),
                    "body": _StationaryBody(),
                    "target_position": position,
                    "initial_pose": initial_pose,
                    "initial_z": 0.765,
                    "active_side": "left",
                    "activation_step": index * 100,
                    "close_start_step": index * 100 + 10,
                    "closed_step": index * 100 + 20,
                    "release_start_step": index * 100 + 70,
                    "open_step": index * 100 + 80,
                    "transport_reference": np.eye(4),
                    "max_z": 0.815,
                    "max_linear_speed": 0.1,
                    "max_angular_speed": 0.2,
                    "max_step_displacement": 0.001,
                    "max_transport_drift": 0.001,
                    "max_transport_rotation": 0.01,
                    "finger_contact_frames": {"left_finger": 10, "right_finger": 10},
                    "bilateral_contact_frames": 10,
                    "table_contact_frames": 20,
                    "table_contact_while_lifted_frames": 0,
                    "final_table_contact": True,
                }
            )
        task = type("Task", (), {"check_success": lambda self: True})()

        passing = rule.finalize(task)
        self.assertTrue(passing["physical_validation_passed"])

        rule.blocks[1]["bilateral_contact_frames"] = 0
        failing = rule.finalize(task)
        self.assertFalse(failing["physical_validation_passed"])
        self.assertFalse(failing["criteria"]["all_blocks_grasp_contact_success"])


if __name__ == "__main__":
    unittest.main()
