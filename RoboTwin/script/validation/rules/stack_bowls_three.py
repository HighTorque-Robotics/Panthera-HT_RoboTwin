"""Contact and motion diagnostics for the stack_bowls_three task.

This rule deliberately records evidence only.  It does not alter control,
collision filtering, or task-success semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .move_pillbottle_pad import (
    _body_entity,
    _contact_metrics,
    _dynamic_component,
    _entity,
    _entity_id,
    _pose_matrix,
)


class StackBowlsThreeRule:
    """Record bowl motion and all physical bowl contact categories."""

    PHYSICAL_IMPULSE_EPS = 1e-8
    TRACE_STRIDE_STEPS = 25

    def __init__(self):
        self.started = False

    def start(self, task: Any) -> None:
        if self.started:
            raise RuntimeError("stack_bowls_three rule has already started")

        self.dt = float(task.scene.get_timestep())
        self.table_entity = _entity(task.table)
        self.table_id = _entity_id(self.table_entity)
        self.robot_links = self._robot_link_index(task)
        self.robot_link_names = self._robot_link_name_index(task)
        self.bowls = []
        for label, actor in zip(("bowl1", "bowl2", "bowl3"), task.bowl_actors):
            pose = _pose_matrix(actor.get_pose())
            entity = _entity(actor)
            self.bowls.append(
                {
                    "label": label,
                    "actor": actor,
                    "entity": entity,
                    "entity_id": _entity_id(entity),
                    "body": _dynamic_component(entity),
                    "initial_position": pose[:3, 3].copy(),
                    "previous_position": pose[:3, 3].copy(),
                    "max_linear_speed": 0.0,
                    "max_angular_speed": 0.0,
                    "max_step_displacement": 0.0,
                    "max_z": float(pose[2, 3]),
                }
            )
        self.bowl_by_id = {bowl["entity_id"]: bowl for bowl in self.bowls}
        self.contact_pairs = {}
        self.physical_contact_events = []
        self.previous_physical_pairs = set()
        self.trace = []
        self.observed_steps = 0
        self.started = True

    @staticmethod
    def _robot_articulations(task: Any):
        if getattr(task, "single_arm_mode", False):
            return (("left", task.robot.left_entity),)
        return (
            ("left", task.robot.left_entity),
            ("right", task.robot.right_entity),
        )

    @classmethod
    def _robot_link_index(cls, task: Any):
        result = {}
        for side, articulation in cls._robot_articulations(task):
            for link in articulation.get_links():
                if link.get_collision_shapes():
                    result[_entity_id(link.entity)] = (side, link.get_name())
        return result

    @classmethod
    def _robot_link_name_index(cls, task: Any):
        result = {}
        for side, articulation in cls._robot_articulations(task):
            for link in articulation.get_links():
                if link.get_collision_shapes():
                    result.setdefault(link.get_name(), []).append(
                        (side, link.get_name())
                    )
        return result

    def _resolve_robot_link(self, entity):
        link = self.robot_links.get(_entity_id(entity))
        if link is not None:
            return link
        candidates = self.robot_link_names.get(entity.get_name(), [])
        return candidates[0] if candidates else None

    def _semantic_name(self, entity) -> str:
        entity_id = _entity_id(entity)
        bowl = self.bowl_by_id.get(entity_id)
        if bowl is not None:
            return bowl["label"]
        if entity_id == self.table_id:
            return "table"
        robot_link = self._resolve_robot_link(entity)
        if robot_link is not None:
            return f"robot:{robot_link[0]}:{robot_link[1]}"
        return str(entity.get_name())

    @staticmethod
    def _contact_category(entity_a, entity_b, bowl_by_id, table_id, resolve_robot):
        id_a, id_b = _entity_id(entity_a), _entity_id(entity_b)
        bowl_a, bowl_b = id_a in bowl_by_id, id_b in bowl_by_id
        if bowl_a and bowl_b:
            return "bowl_bowl"
        if bowl_a or bowl_b:
            other = entity_b if bowl_a else entity_a
            other_id = id_b if bowl_a else id_a
            if other_id == table_id:
                return "bowl_table"
            if resolve_robot(other) is not None:
                return "bowl_robot"
        return None

    def _observe_bowls(self):
        positions = {}
        for bowl in self.bowls:
            pose = _pose_matrix(bowl["actor"].get_pose())
            position = pose[:3, 3]
            bowl["max_z"] = max(bowl["max_z"], float(position[2]))
            bowl["max_step_displacement"] = max(
                bowl["max_step_displacement"],
                float(np.linalg.norm(position - bowl["previous_position"])),
            )
            bowl["previous_position"] = position.copy()
            bowl["max_linear_speed"] = max(
                bowl["max_linear_speed"],
                float(np.linalg.norm(bowl["body"].get_linear_velocity())),
            )
            bowl["max_angular_speed"] = max(
                bowl["max_angular_speed"],
                float(np.linalg.norm(bowl["body"].get_angular_velocity())),
            )
            positions[bowl["label"]] = position.tolist()
        return positions

    def _observe_contacts(self, task: Any, step_index: int):
        current_physical_pairs = set()
        for contact in task.scene.get_contacts():
            if len(contact.bodies) != 2 or not contact.points:
                continue
            entity_a = _body_entity(contact.bodies[0])
            entity_b = _body_entity(contact.bodies[1])
            if entity_a is None or entity_b is None:
                continue
            separation, impulse = _contact_metrics(contact)
            physical = separation <= 0.0 or impulse > self.PHYSICAL_IMPULSE_EPS
            name_a = self._semantic_name(entity_a)
            name_b = self._semantic_name(entity_b)
            pair = "-".join(sorted((name_a, name_b)))
            record = self.contact_pairs.setdefault(
                pair,
                {
                    "category": self._contact_category(
                        entity_a,
                        entity_b,
                        self.bowl_by_id,
                        self.table_id,
                        self._resolve_robot_link,
                    ),
                    "frames": 0,
                    "physical_frames": 0,
                    "first_step": step_index,
                    "first_time_s": step_index * self.dt,
                    "minimum_separation_m": float("inf"),
                    "maximum_impulse": 0.0,
                },
            )
            record["frames"] += 1
            record["physical_frames"] += int(physical)
            record["minimum_separation_m"] = min(
                record["minimum_separation_m"], separation
            )
            record["maximum_impulse"] = max(record["maximum_impulse"], impulse)

            category = record["category"]
            if physical and category in {"bowl_bowl", "bowl_table", "bowl_robot"}:
                if (
                    pair not in self.previous_physical_pairs
                    and pair not in current_physical_pairs
                ):
                    self.physical_contact_events.append(
                        {
                            "step": step_index,
                            "time_s": step_index * self.dt,
                            "policy_step": int(getattr(task, "take_action_cnt", 0)),
                            "category": category,
                            "pair": pair,
                            "separation_m": separation,
                            "impulse_Ns": impulse,
                        }
                    )
                current_physical_pairs.add(pair)
        self.previous_physical_pairs = current_physical_pairs

    def observe(self, task: Any, step_index: int) -> None:
        if not self.started:
            raise RuntimeError("Rule must be started before observation")
        positions = self._observe_bowls()
        self._observe_contacts(task, step_index)
        if step_index % self.TRACE_STRIDE_STEPS == 0:
            self.trace.append(
                {
                    "step": step_index,
                    "time_s": step_index * self.dt,
                    "policy_step": int(getattr(task, "take_action_cnt", 0)),
                    "bowl_positions": positions,
                    "left_gripper": float(task.robot.get_left_gripper_val()),
                    "right_gripper": float(task.robot.get_right_gripper_val()),
                }
            )
        self.observed_steps += 1

    def finalize(self, task: Any) -> dict[str, Any]:
        bowl_reports = []
        for bowl in self.bowls:
            final_pose = _pose_matrix(bowl["actor"].get_pose())
            final_speed = float(np.linalg.norm(bowl["body"].get_linear_velocity()))
            bowl_reports.append(
                {
                    "label": bowl["label"],
                    "metrics": {
                        "initial_position": bowl["initial_position"].tolist(),
                        "final_position": final_pose[:3, 3].tolist(),
                        "maximum_z_m": bowl["max_z"],
                        "maximum_linear_speed_m_s": bowl["max_linear_speed"],
                        "maximum_angular_speed_rad_s": bowl["max_angular_speed"],
                        "maximum_step_displacement_m": bowl["max_step_displacement"],
                        "final_linear_speed_m_s": final_speed,
                    },
                }
            )

        categories = {"bowl_bowl": [], "bowl_table": [], "bowl_robot": []}
        for pair, record in self.contact_pairs.items():
            if record["category"] in categories:
                categories[record["category"]].append({"pair": pair, **record})

        # This rule is evidence collection, not a task-success validator.
        return {
            "schema_version": 1,
            "task": "stack_bowls_three",
            "diagnostic_only": True,
            "sample_dt_s": self.dt,
            "threshold_status": "not_applicable",
            "observed_physics_steps": self.observed_steps,
            "bowl_reports": bowl_reports,
            "contact_pairs": categories,
            "physical_contact_events": self.physical_contact_events,
            "criteria": {"telemetry_collected": self.observed_steps > 0},
            "telemetry_collected": bool(self.observed_steps > 0),
            "task_success": bool(task.check_success()),
            "trace_stride_steps": self.TRACE_STRIDE_STEPS,
            "trace": self.trace,
        }
