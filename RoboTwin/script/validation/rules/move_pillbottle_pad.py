"""Physics-quality rules for the move_pillbottle_pad task."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien.physx as sapien_physx


def _pose_matrix(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=float)


def _entity(actor_or_entity):
    return getattr(actor_or_entity, "actor", actor_or_entity)


def _body_entity(body):
    return getattr(body, "entity", None)


def _body_name(body) -> str:
    entity = _body_entity(body)
    if entity is None:
        return type(body).__name__
    return str(entity.get_name())


def _entity_id(entity) -> int:
    """Return SAPIEN's stable ID instead of a transient Python wrapper ID."""
    return int(entity.get_global_id())


def _contact_metrics(contact) -> tuple[float, float]:
    separations = [float(point.separation) for point in contact.points]
    impulses = [float(np.linalg.norm(point.impulse)) for point in contact.points]
    return (
        min(separations, default=float("inf")),
        max(impulses, default=0.0),
    )


def _rotation_distance(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _dynamic_component(entity):
    for component in entity.get_components():
        if isinstance(component, sapien_physx.PhysxRigidDynamicComponent):
            return component
    raise TypeError(f"Entity {entity.get_name()!r} has no dynamic rigid body")


class MovePillbottlePadRule:
    """Observe grasp, lift, transport, and release of the pill bottle."""

    MIN_LIFT_M = 0.03
    MIN_BILATERAL_CONTACT_S = 0.02
    MAX_OBJECT_EE_DRIFT_M = 0.02
    MAX_OBJECT_EE_ROTATION_RAD = 0.35
    MAX_PENETRATION_M = 0.001
    MAX_FINAL_LINEAR_SPEED_M_S = 0.05
    PHYSICAL_IMPULSE_EPS = 1e-8

    def __init__(self):
        self.started = False

    def start(self, task: Any) -> None:
        if self.started:
            raise RuntimeError("move_pillbottle_pad rule has already started")

        self.dt = float(task.scene.get_timestep())
        self.object_entity = _entity(task.pillbottle)
        self.table_entity = _entity(task.table)
        self.pad_entity = _entity(task.pad)
        self.object_body = _dynamic_component(self.object_entity)

        initial_pose = _pose_matrix(task.pillbottle.get_pose())
        self.initial_object_pose = initial_pose
        self.initial_object_z = float(initial_pose[2, 3])
        self.single_arm_mode = bool(getattr(task, "single_arm_mode", False))
        self.active_side = getattr(
            task,
            "active_arm_side",
            "right" if initial_pose[0, 3] > 0 else "left",
        )
        self.robot_links = self._robot_link_index(task)
        self.robot_link_names = self._robot_link_name_index(task)
        self.active_finger_names = {
            joint[0].child_link.get_name()
            for joint in (
                task.robot.left_gripper
                if self.active_side == "left"
                else task.robot.right_gripper
            )
        }

        self.previous_gripper = self._gripper_value(task)
        self.close_start_step = None
        self.closed_step = None
        self.release_start_step = None
        self.open_step = None
        self.transport_reference = None

        self.max_object_z = self.initial_object_z
        self.max_object_linear_speed = 0.0
        self.max_object_angular_speed = 0.0
        self.max_object_step_displacement = 0.0
        self.previous_object_position = initial_pose[:3, 3].copy()
        self.max_transport_drift = 0.0
        self.max_transport_rotation = 0.0

        self.finger_contact_frames = {name: 0 for name in self.active_finger_names}
        self.bilateral_contact_frames = 0
        self.table_contact_frames = 0
        self.table_contact_while_lifted_frames = 0
        self.pad_contact_frames = 0
        self.final_table_contact = False
        self.final_pad_contact = False

        self.minimum_robot_separation = float("inf")
        self.maximum_robot_contact_impulse = 0.0
        self.maximum_robot_penetration = 0.0
        self.unexpected_robot_contacts = {}
        self.self_contacts = {}
        self.contact_pairs = {}
        self.trace = []
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
    def _robot_link_index(cls, task: Any) -> dict[int, tuple[str, str]]:
        result = {}
        for side, articulation in cls._robot_articulations(task):
            for link in articulation.get_links():
                if not link.get_collision_shapes():
                    continue
                result[_entity_id(link.entity)] = (side, link.get_name())
        return result

    @classmethod
    def _robot_link_name_index(cls, task: Any) -> dict[str, list[tuple[str, str]]]:
        result = {}
        for side, articulation in cls._robot_articulations(task):
            for link in articulation.get_links():
                if not link.get_collision_shapes():
                    continue
                result.setdefault(link.get_name(), []).append((side, link.get_name()))
        return result

    def _resolve_robot_link(self, entity):
        link = self.robot_links.get(_entity_id(entity))
        if link is not None:
            return link
        candidates = self.robot_link_names.get(entity.get_name(), [])
        for candidate in candidates:
            if candidate[0] == self.active_side:
                return candidate
        return candidates[0] if candidates else None

    def _gripper_value(self, task: Any) -> float:
        if self.active_side == "left":
            return float(task.robot.get_left_gripper_val())
        return float(task.robot.get_right_gripper_val())

    def _end_effector_matrix(self, task: Any) -> np.ndarray:
        joint = task.robot.left_ee if self.active_side == "left" else task.robot.right_ee
        return _pose_matrix(joint.child_link.get_entity_pose())

    @staticmethod
    def _update_pair_record(records, pair, step_index, separation, impulse):
        record = records.setdefault(
            pair,
            {
                "first_step": step_index,
                "frames": 0,
                "minimum_separation_m": float("inf"),
                "maximum_impulse": 0.0,
            },
        )
        record["frames"] += 1
        record["minimum_separation_m"] = min(
            record["minimum_separation_m"], separation
        )
        record["maximum_impulse"] = max(record["maximum_impulse"], impulse)

    def _observe_contacts(self, task: Any, step_index: int):
        object_id = _entity_id(self.object_entity)
        table_id = _entity_id(self.table_entity)
        pad_id = _entity_id(self.pad_entity)
        finger_contacts = set()
        table_contact = False
        pad_contact = False

        for contact in task.scene.get_contacts():
            if len(contact.bodies) != 2 or not contact.points:
                continue
            entity_a = _body_entity(contact.bodies[0])
            entity_b = _body_entity(contact.bodies[1])
            if entity_a is None or entity_b is None:
                continue
            id_a, id_b = _entity_id(entity_a), _entity_id(entity_b)
            separation, impulse = _contact_metrics(contact)
            physical = separation <= 0.0 or impulse > self.PHYSICAL_IMPULSE_EPS
            pair_name = "-".join(sorted((_body_name(contact.bodies[0]), _body_name(contact.bodies[1]))))
            pair_record = self.contact_pairs.setdefault(
                pair_name,
                {
                    "frames": 0,
                    "first_step": step_index,
                    "minimum_separation_m": float("inf"),
                    "maximum_impulse": 0.0,
                    "physical_frames": 0,
                },
            )
            pair_record["frames"] += 1
            pair_record["minimum_separation_m"] = min(
                pair_record["minimum_separation_m"], separation
            )
            pair_record["maximum_impulse"] = max(
                pair_record["maximum_impulse"], impulse
            )
            pair_record["physical_frames"] += int(physical)

            if object_id in {id_a, id_b}:
                other_entity = entity_b if id_a == object_id else entity_a
                other_id = _entity_id(other_entity)
                robot_link = self._resolve_robot_link(other_entity)
                if robot_link is not None:
                    side, link_name = robot_link
                    if physical and side == self.active_side and link_name in self.active_finger_names:
                        finger_contacts.add(link_name)
                elif physical and other_id == table_id:
                    table_contact = True
                elif physical and other_id == pad_id:
                    pad_contact = True

            robot_a = self._resolve_robot_link(entity_a)
            robot_b = self._resolve_robot_link(entity_b)
            if robot_a is not None and robot_b is not None:
                if robot_a[0] == robot_b[0]:
                    pair = f"{robot_a[0]}:{'-'.join(sorted((robot_a[1], robot_b[1])))}"
                    self._update_pair_record(
                        self.self_contacts, pair, step_index, separation, impulse
                    )
                elif physical:
                    pair = f"cross_arm:{robot_a[1]}-{robot_b[1]}"
                    self._update_pair_record(
                        self.unexpected_robot_contacts,
                        pair,
                        step_index,
                        separation,
                        impulse,
                    )

            if robot_a is None and robot_b is None:
                continue
            robot = robot_a if robot_a is not None else robot_b
            other_id = id_b if robot_a is not None else id_a
            expected_grasp = (
                other_id == object_id
                and robot[0] == self.active_side
                and robot[1] in self.active_finger_names
            )
            supported_base = robot[1] == "base_link" and other_id == table_id
            if physical and not expected_grasp and not supported_base:
                other_name = _body_name(
                    contact.bodies[1] if robot_a is not None else contact.bodies[0]
                )
                pair = f"{robot[0]}:{robot[1]}-{other_name}"
                self._update_pair_record(
                    self.unexpected_robot_contacts,
                    pair,
                    step_index,
                    separation,
                    impulse,
                )

            if robot_a is not None or robot_b is not None:
                self.minimum_robot_separation = min(
                    self.minimum_robot_separation, separation
                )
                self.maximum_robot_contact_impulse = max(
                    self.maximum_robot_contact_impulse, impulse
                )
                self.maximum_robot_penetration = max(
                    self.maximum_robot_penetration, max(0.0, -separation)
                )

        for name in finger_contacts:
            self.finger_contact_frames[name] += 1
        if self.active_finger_names.issubset(finger_contacts):
            self.bilateral_contact_frames += 1
        if table_contact:
            self.table_contact_frames += 1
        if pad_contact:
            self.pad_contact_frames += 1
        self.final_table_contact = table_contact
        self.final_pad_contact = pad_contact
        return finger_contacts, table_contact, pad_contact

    def observe(self, task: Any, step_index: int) -> None:
        if not self.started:
            raise RuntimeError("Rule must be started before observation")

        gripper = self._gripper_value(task)
        if self.close_start_step is None and gripper < 0.95:
            self.close_start_step = step_index
        if self.closed_step is None and gripper <= 0.2:
            self.closed_step = step_index
        if (
            self.closed_step is not None
            and self.release_start_step is None
            and gripper > self.previous_gripper + 1e-6
        ):
            self.release_start_step = step_index
        if self.release_start_step is not None and self.open_step is None and gripper >= 0.8:
            self.open_step = step_index

        object_pose = _pose_matrix(task.pillbottle.get_pose())
        object_position = object_pose[:3, 3]
        self.max_object_z = max(self.max_object_z, float(object_position[2]))
        self.max_object_step_displacement = max(
            self.max_object_step_displacement,
            float(np.linalg.norm(object_position - self.previous_object_position)),
        )
        self.previous_object_position = object_position.copy()
        self.max_object_linear_speed = max(
            self.max_object_linear_speed,
            float(np.linalg.norm(self.object_body.get_linear_velocity())),
        )
        self.max_object_angular_speed = max(
            self.max_object_angular_speed,
            float(np.linalg.norm(self.object_body.get_angular_velocity())),
        )

        finger_contacts, table_contact, pad_contact = self._observe_contacts(
            task, step_index
        )
        lifted = object_position[2] >= self.initial_object_z + self.MIN_LIFT_M
        if lifted and table_contact:
            self.table_contact_while_lifted_frames += 1

        ee_pose = self._end_effector_matrix(task)
        relative_pose = np.linalg.inv(ee_pose) @ object_pose
        if (
            self.transport_reference is None
            and self.closed_step is not None
            and self.active_finger_names.issubset(finger_contacts)
        ):
            self.transport_reference = relative_pose.copy()
        if self.transport_reference is not None and self.release_start_step is None:
            drift = float(
                np.linalg.norm(
                    relative_pose[:3, 3] - self.transport_reference[:3, 3]
                )
            )
            rotation = _rotation_distance(
                self.transport_reference[:3, :3], relative_pose[:3, :3]
            )
            self.max_transport_drift = max(self.max_transport_drift, drift)
            self.max_transport_rotation = max(self.max_transport_rotation, rotation)

        if step_index % 25 == 0:
            self.trace.append(
                {
                    "step": step_index,
                    "time_s": step_index * self.dt,
                    "gripper": gripper,
                    "object_position": object_position.tolist(),
                    "finger_contacts": sorted(finger_contacts),
                    "table_contact": table_contact,
                    "pad_contact": pad_contact,
                }
            )
        self.previous_gripper = gripper

    def finalize(self, task: Any) -> dict[str, Any]:
        final_pose = _pose_matrix(task.pillbottle.get_pose())
        final_position = final_pose[:3, 3]
        target_position = np.asarray(task.pad.get_pose().p, dtype=float)
        final_linear_speed = float(np.linalg.norm(self.object_body.get_linear_velocity()))

        bilateral_duration = self.bilateral_contact_frames * self.dt
        object_lift = self.max_object_z - self.initial_object_z
        grasp_contact_success = (
            bilateral_duration >= self.MIN_BILATERAL_CONTACT_S
        )
        lift_success = (
            object_lift >= self.MIN_LIFT_M
            and self.table_contact_while_lifted_frames == 0
        )
        object_follow_success = (
            self.transport_reference is not None
            and self.max_transport_drift <= self.MAX_OBJECT_EE_DRIFT_M
            and self.max_transport_rotation <= self.MAX_OBJECT_EE_ROTATION_RAD
        )
        collision_safe = (
            self.maximum_robot_penetration <= self.MAX_PENETRATION_M
            and not self.unexpected_robot_contacts
        )
        release_success = (
            self.release_start_step is not None
            and self.open_step is not None
            and (self.final_table_contact or self.final_pad_contact)
            and final_linear_speed <= self.MAX_FINAL_LINEAR_SPEED_M_S
        )
        task_success = bool(task.check_success())
        criteria = {
            "grasp_contact_success": grasp_contact_success,
            "lift_success": lift_success,
            "object_follow_success": object_follow_success,
            "collision_safe": collision_safe,
            "release_success": release_success,
            "task_success": task_success,
        }

        return {
            "schema_version": 1,
            "task": "move_pillbottle_pad",
            **(
                {"active_arm": "arm"}
                if self.single_arm_mode
                else {"active_side": self.active_side}
            ),
            "sample_dt_s": self.dt,
            "threshold_status": "provisional",
            "thresholds": {
                "minimum_lift_m": self.MIN_LIFT_M,
                "minimum_bilateral_contact_s": self.MIN_BILATERAL_CONTACT_S,
                "maximum_object_ee_drift_m": self.MAX_OBJECT_EE_DRIFT_M,
                "maximum_object_ee_rotation_rad": self.MAX_OBJECT_EE_ROTATION_RAD,
                "maximum_penetration_m": self.MAX_PENETRATION_M,
                "maximum_final_linear_speed_m_s": self.MAX_FINAL_LINEAR_SPEED_M_S,
            },
            "events": {
                "close_start_step": self.close_start_step,
                "closed_step": self.closed_step,
                "release_start_step": self.release_start_step,
                "open_step": self.open_step,
            },
            "metrics": {
                "initial_object_position": self.initial_object_pose[:3, 3].tolist(),
                "final_object_position": final_position.tolist(),
                "target_position": target_position.tolist(),
                "final_xy_error_m": np.abs(
                    final_position[:2] - target_position[:2]
                ).tolist(),
                "object_lift_m": object_lift,
                "maximum_object_ee_drift_m": self.max_transport_drift,
                "maximum_object_ee_rotation_rad": self.max_transport_rotation,
                "maximum_object_linear_speed_m_s": self.max_object_linear_speed,
                "maximum_object_angular_speed_rad_s": self.max_object_angular_speed,
                "maximum_object_step_displacement_m": self.max_object_step_displacement,
                "final_object_linear_speed_m_s": final_linear_speed,
                "finger_contact_frames": self.finger_contact_frames,
                "bilateral_contact_frames": self.bilateral_contact_frames,
                "bilateral_contact_duration_s": bilateral_duration,
                "table_contact_frames": self.table_contact_frames,
                "table_contact_while_lifted_frames": self.table_contact_while_lifted_frames,
                "pad_contact_frames": self.pad_contact_frames,
                "minimum_robot_contact_separation_m": (
                    None
                    if not np.isfinite(self.minimum_robot_separation)
                    else self.minimum_robot_separation
                ),
                "maximum_robot_contact_impulse": self.maximum_robot_contact_impulse,
                "maximum_robot_penetration_m": self.maximum_robot_penetration,
                "self_contact_pairs": self.self_contacts,
                "unexpected_robot_contact_pairs": self.unexpected_robot_contacts,
                "raw_contact_pairs": self.contact_pairs,
            },
            "criteria": criteria,
            "physical_validation_passed": bool(all(criteria.values())),
            "trace_stride_steps": 25,
            "trace": self.trace,
        }
