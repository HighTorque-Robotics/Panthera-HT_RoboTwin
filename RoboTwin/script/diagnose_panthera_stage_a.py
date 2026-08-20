import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_CONFIG = "move_pillbottle_pad_panthera"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_task_args(
    task_config: str,
    seed: int,
    embodiment_override: str | None = None,
    embodiment_distance: float = 0.60,
) -> dict:
    args = load_yaml(ROOT / "task_config" / f"{task_config}.yml")
    registry = load_yaml(ROOT / "task_config" / "_embodiment_config.yml")
    embodiment = args["embodiment"]
    if embodiment_override is not None:
        embodiment = [embodiment_override, embodiment_override, embodiment_distance]

    if len(embodiment) != 3:
        raise ValueError("Stage A expects two independent single-arm embodiments")

    left_name, right_name, embodiment_dis = embodiment
    args.update({
        "task_name": "move_pillbottle_pad",
        "task_config": task_config,
        "left_robot_file": registry[left_name]["file_path"],
        "right_robot_file": registry[right_name]["file_path"],
        "left_embodiment_config": load_yaml(ROOT / registry[left_name]["file_path"] / "config.yml"),
        "right_embodiment_config": load_yaml(ROOT / registry[right_name]["file_path"] / "config.yml"),
        "embodiment_dis": embodiment_dis,
        "dual_arm_embodied": False,
        "embodiment_name": f"{left_name}+{right_name}",
        "need_plan": True,
        "save_data": False,
        "collect_data": False,
        "render_freq": 0,
        "seed": seed,
        "now_ep_num": 0,
    })
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panthera Stage A planning diagnosis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--embodiment-override")
    parser.add_argument("--embodiment-distance", type=float, default=0.60)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=250)
    parser.add_argument("--preserve-urdf-mass", action="store_true")
    parser.add_argument(
        "--filter-curobo-self-collisions",
        action="store_true",
        help="Apply CuRobo self_collision_ignore pairs to SAPIEN collision groups",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def arm_state(task, arm: str) -> dict:
    robot = task.robot
    entity = robot.left_entity if arm == "left" else robot.right_entity
    arm_joints = robot.left_arm_joints if arm == "left" else robot.right_arm_joints
    active_joints = entity.get_active_joints()
    indices = [active_joints.index(joint) for joint in arm_joints]
    qpos = entity.get_qpos()
    qvel = entity.get_qvel()
    return {
        "qpos": np.asarray(qpos)[indices].tolist(),
        "qvel": np.asarray(qvel)[indices].tolist(),
    }


def entity_qpos(task, arm: str) -> np.ndarray:
    entity = task.robot.left_entity if arm == "left" else task.robot.right_entity
    return np.asarray(entity.get_qpos()).copy()


def arm_limits(task, arm: str) -> tuple[np.ndarray, np.ndarray]:
    robot = task.robot
    joints = robot.left_arm_joints if arm == "left" else robot.right_arm_joints
    limits = np.asarray([joint.get_limits()[0] for joint in joints], dtype=float)
    return limits[:, 0], limits[:, 1]


def arm_limit_report(task, arm: str, qpos: np.ndarray) -> dict:
    lower, upper = arm_limits(task, arm)
    margin = np.minimum(qpos - lower, upper - qpos)
    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "margin": margin.tolist(),
        "violates_limit": bool(np.any(margin < 0)),
    }


def with_arm_qpos(task, arm: str, full_qpos: np.ndarray, arm_qpos: np.ndarray) -> np.ndarray:
    robot = task.robot
    entity = robot.left_entity if arm == "left" else robot.right_entity
    arm_joints = robot.left_arm_joints if arm == "left" else robot.right_arm_joints
    active_joints = entity.get_active_joints()
    result = full_qpos.copy()
    for joint, value in zip(arm_joints, arm_qpos):
        result[active_joints.index(joint)] = value
    return result


def replace_selected_joints(actual_qpos: np.ndarray, planned_qpos: np.ndarray, indices: list[int]) -> np.ndarray:
    result = actual_qpos.copy()
    result[indices] = planned_qpos[indices]
    return result


def summarize_plan(result: dict) -> dict:
    summary = {"status": result.get("status", "Missing")}
    if summary["status"] != "Success":
        return summary

    position = np.asarray(result["position"])
    velocity = np.asarray(result["velocity"])
    summary.update({
        "steps": int(position.shape[0]),
        "end_qpos": position[-1].tolist(),
        "end_qvel": velocity[-1].tolist(),
        "max_abs_qvel": np.max(np.abs(velocity), axis=0).tolist(),
    })
    return summary


def repeated_plans(plan_func, target_pose: list, constraint_pose: list, start_qpos: np.ndarray, repeat: int) -> list:
    return [
        summarize_plan(
            plan_func(
                target_pose,
                constraint_pose=constraint_pose,
                last_qpos=start_qpos,
            ))
        for _ in range(repeat)
    ]


def settle_physics(task, steps: int) -> None:
    for _ in range(steps):
        task.robot._entity_qf(task.robot.left_entity)
        task.robot._entity_qf(task.robot.right_entity)
        task.scene.step()


def scene_contact_summary(task) -> list[dict]:
    pairs = {}
    for contact in task.scene.get_contacts():
        body_infos = []
        for body in contact.bodies:
            entity = body.entity
            pose = entity.get_pose()
            body_infos.append({
                "name": entity.get_name(),
                "pose": pose.p.tolist() + pose.q.tolist(),
            })
        key = tuple(sorted(
            (info["name"], tuple(round(value, 5) for value in info["pose"]))
            for info in body_infos
        ))
        if key not in pairs:
            pairs[key] = {
                "bodies": [name for name, _ in key],
                "body_poses": [list(pose) for _, pose in key],
                "point_count": 0,
            }
        pairs[key]["point_count"] += len(contact.points)
    return [pairs[key] for key in sorted(pairs)]


def apply_curobo_self_collision_filter(robot) -> None:
    """Match SAPIEN robot self-collision filtering to each CuRobo config."""
    for arm_index, (entity, config_path) in enumerate((
        (robot.left_entity, robot.left_curobo_yml_path),
        (robot.right_entity, robot.right_curobo_yml_path),
    )):
        config = load_yaml(ROOT / config_path)
        ignore_map = config["robot_cfg"]["kinematics"]["self_collision_ignore"]
        ignored_pairs = sorted({
            tuple(sorted((link_name, ignored_name)))
            for link_name, ignored_names in ignore_map.items()
            for ignored_name in ignored_names
        })
        if len(ignored_pairs) > 32:
            raise ValueError(f"SAPIEN collision groups support at most 32 pairs, got {len(ignored_pairs)}")

        masks = {link_name: 0 for link_name in ignore_map}
        for bit, (link_a, link_b) in enumerate(ignored_pairs):
            mask = 1 << bit
            masks.setdefault(link_a, 0)
            masks.setdefault(link_b, 0)
            masks[link_a] |= mask
            masks[link_b] |= mask

        link_by_name = {link.get_name(): link for link in entity.get_links()}
        articulation_id = 0x400 + arm_index
        for link_name, mask in masks.items():
            if link_name not in link_by_name:
                raise ValueError(f"CuRobo collision link {link_name!r} is missing from SAPIEN articulation")
            for shape in link_by_name[link_name].get_collision_shapes():
                groups = list(shape.get_collision_groups())
                groups[2] |= mask
                groups[3] = articulation_id
                shape.set_collision_groups(groups)


def action_to_dict(action) -> dict:
    result = {
        "arm": str(action.arm_tag),
        "action": action.action,
        "args": action.args,
    }
    if action.action == "move":
        result["target_pose"] = action.target_pose
    else:
        result["target_gripper_pos"] = action.target_gripper_pos
    return result


def probe_scene(
    task_args: dict,
    repeat: int,
    settle_steps: int,
    preserve_urdf_mass: bool,
    filter_curobo_self_collisions: bool,
) -> dict:
    sys.path.insert(0, str(ROOT))
    from envs.move_pillbottle_pad import move_pillbottle_pad
    from envs.robot import Robot
    from envs.utils import ArmTag

    class DiagnosticTask(move_pillbottle_pad):

        def load_robot(self, **kwargs):
            if not hasattr(self, "robot"):
                self.robot = Robot(self.scene, self.need_topp, **kwargs)
                if filter_curobo_self_collisions:
                    apply_curobo_self_collision_filter(self.robot)
                self.robot.set_planner(self.scene)
                self.robot.init_joints()
            else:
                self.robot.reset(self.scene, self.need_topp, **kwargs)
                if filter_curobo_self_collisions:
                    apply_curobo_self_collision_filter(self.robot)

            if not preserve_urdf_mass:
                for link in self.robot.left_entity.get_links():
                    link.set_mass(1)
                for link in self.robot.right_entity.get_links():
                    link.set_mass(1)

    task = DiagnosticTask()
    try:
        task.setup_demo(**task_args)
        pill_pose = task.pillbottle.get_pose()
        pad_pose = task.pad.get_pose()
        arm = "right" if pill_pose.p[0] > 0 else "left"
        initial_full_qpos = entity_qpos(task, arm)
        initial_state = {
            "left": arm_state(task, "left"),
            "right": arm_state(task, "right"),
        }
        robot_root_poses = {
            "left": task.robot.left_entity.get_root_pose().p.tolist() + task.robot.left_entity.get_root_pose().q.tolist(),
            "right": task.robot.right_entity.get_root_pose().p.tolist() + task.robot.right_entity.get_root_pose().q.tolist(),
        }
        initial_contacts = scene_contact_summary(task)
        contacts = [
            {
                "id": index,
                "pose": contact_pose,
            }
            for index, contact_pose in task.pillbottle.iter_contact_points("list")
        ]
        selected_arm, actions = task.grasp_actor(
            task.pillbottle,
            arm_tag=ArmTag(arm),
            pre_grasp_dis=0.06,
            gripper_pos=0,
        )
        move_actions = [action for action in actions if action.action == "move"]
        if len(move_actions) != 2:
            raise RuntimeError(f"Expected two grasp move actions, got {len(move_actions)}")

        pre_action, contact_action = move_actions
        plan_func = task.robot.left_plan_path if arm == "left" else task.robot.right_plan_path
        pre_result = plan_func(pre_action.target_pose)
        pre_summary = summarize_plan(pre_result)

        planning = {
            "pregrasp": pre_summary,
            "contact_constraint": contact_action.args["constraint_pose"],
        }
        if pre_summary["status"] == "Success":
            planned_end_full_qpos = with_arm_qpos(
                task,
                arm,
                initial_full_qpos,
                np.asarray(pre_result["position"][-1]),
            )
            control_seq = {
                "left_arm": pre_result if arm == "left" else None,
                "left_gripper": None,
                "right_arm": pre_result if arm == "right" else None,
                "right_gripper": None,
            }
            task.take_dense_action(control_seq, save_freq=None)
            contacts_after_pregrasp = scene_contact_summary(task)
            actual_full_qpos = entity_qpos(task, arm)
            actual_arm_qpos = np.asarray(arm_state(task, arm)["qpos"], dtype=float)
            planned_arm_qpos = np.asarray(pre_result["position"][-1], dtype=float)
            lower, upper = arm_limits(task, arm)
            clipped_arm_qpos = np.clip(actual_arm_qpos, lower + 1e-5, upper - 1e-5)
            clipped_full_qpos = with_arm_qpos(
                task,
                arm,
                actual_full_qpos,
                clipped_arm_qpos,
            )
            joint_count = planned_arm_qpos.shape[0]
            sensitivity_variants = {
                "actual_with_joint4_from_planned": [3],
                "actual_with_joint5_from_planned": [4],
                "actual_with_joint6_from_planned": [5],
                f"actual_with_wrist_joints4_to{joint_count}_from_planned": list(range(3, joint_count)),
                "actual_with_shoulder_joints2_and3_from_planned": [1, 2],
            }
            sensitivity_results = {}
            for name, indices in sensitivity_variants.items():
                variant_arm_qpos = replace_selected_joints(
                    actual_arm_qpos,
                    planned_arm_qpos,
                    indices,
                )
                variant_full_qpos = with_arm_qpos(
                    task,
                    arm,
                    actual_full_qpos,
                    variant_arm_qpos,
                )
                sensitivity_results[name] = {
                    "qpos": variant_arm_qpos.tolist(),
                    "plans": repeated_plans(
                        plan_func,
                        contact_action.target_pose,
                        contact_action.args["constraint_pose"],
                        variant_full_qpos,
                        repeat,
                    ),
                }

            first_contact = plan_func(
                contact_action.target_pose,
                constraint_pose=contact_action.args["constraint_pose"],
            )
            planning.update({
                "state_after_pregrasp_execution": arm_state(task, arm),
                "contacts_after_pregrasp_execution": contacts_after_pregrasp,
                "state_after_pregrasp_limit_report": arm_limit_report(task, arm, actual_arm_qpos),
                "planned_vs_actual_qpos_delta": (actual_arm_qpos - planned_arm_qpos).tolist(),
                "contact_first_attempt_from_actual": summarize_plan(first_contact),
                "contact_state_sensitivity": sensitivity_results,
                "contact_repeats": {
                    "from_initial_qpos": repeated_plans(
                        plan_func,
                        contact_action.target_pose,
                        contact_action.args["constraint_pose"],
                        initial_full_qpos,
                        repeat,
                    ),
                    "from_planned_pregrasp_end_qpos": repeated_plans(
                        plan_func,
                        contact_action.target_pose,
                        contact_action.args["constraint_pose"],
                        planned_end_full_qpos,
                        repeat,
                    ),
                    "from_actual_pregrasp_end_qpos": repeated_plans(
                        plan_func,
                        contact_action.target_pose,
                        contact_action.args["constraint_pose"],
                        actual_full_qpos,
                        repeat,
                    ),
                    "from_actual_qpos_clipped_to_limits": repeated_plans(
                        plan_func,
                        contact_action.target_pose,
                        contact_action.args["constraint_pose"],
                        clipped_full_qpos,
                        repeat,
                    ),
                },
            })
            settle_physics(task, settle_steps)
            settled_full_qpos = entity_qpos(task, arm)
            settled_arm_state = arm_state(task, arm)
            planning.update({
                "settle_steps": settle_steps,
                "state_after_settle": settled_arm_state,
                "state_after_settle_limit_report": arm_limit_report(
                    task,
                    arm,
                    np.asarray(settled_arm_state["qpos"], dtype=float),
                ),
                "contact_after_settle": repeated_plans(
                    plan_func,
                    contact_action.target_pose,
                    contact_action.args["constraint_pose"],
                    settled_full_qpos,
                    repeat,
                ),
            })
        return {
            "pillbottle_model_id": int(task.pillbottle_id),
            "pillbottle_pose": pill_pose.p.tolist() + pill_pose.q.tolist(),
            "pad_pose": pad_pose.p.tolist() + pad_pose.q.tolist(),
            "selected_arm": str(selected_arm),
            "initial_state": initial_state,
            "robot_root_poses": robot_root_poses,
            "filter_curobo_self_collisions": filter_curobo_self_collisions,
            "initial_contacts": initial_contacts,
            "contact_points": contacts,
            "grasp_actions": [action_to_dict(action) for action in actions],
            "plan_success_after_target_selection": bool(task.plan_success),
            "planning": planning,
        }
    finally:
        task.close_env()


def main() -> None:
    cli = parse_args()
    output = cli.output or Path(f"/tmp/panthera_stage_a_seed{cli.seed}.json")
    task_args = build_task_args(
        cli.task_config,
        cli.seed,
        embodiment_override=cli.embodiment_override,
        embodiment_distance=cli.embodiment_distance,
    )
    summary = {
        "seed": cli.seed,
        "repeat": cli.repeat,
        "settle_steps": cli.settle_steps,
        "preserve_urdf_mass": cli.preserve_urdf_mass,
        "filter_curobo_self_collisions": cli.filter_curobo_self_collisions,
        "task_config": cli.task_config,
        "embodiment": task_args["embodiment_name"],
        "output": str(output),
    }
    if not cli.config_only:
        summary["scene"] = probe_scene(
            task_args,
            cli.repeat,
            cli.settle_steps,
            cli.preserve_urdf_mass,
            cli.filter_curobo_self_collisions,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
