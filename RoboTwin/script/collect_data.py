import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


PANTHERA_SINGLE_ARM_TASKS = frozenset({
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "place_a2b_left",
    "place_a2b_right",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "rotate_qrcode",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
})


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_collection_config(task_name, task_config):
    """Load an explicit config or derive the approved Panthera single-arm view."""
    for suffix in (".yaml", ".yml"):
        config_path = Path("task_config") / f"{task_config}{suffix}"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as file:
                return yaml.load(file.read(), Loader=yaml.FullLoader)

    expected_single_name = f"{task_name}_panthera_single"
    if task_config != expected_single_name or task_name not in PANTHERA_SINGLE_ARM_TASKS:
        raise FileNotFoundError(f"Task config not found: {task_config}.yml")

    base_path = Path("task_config") / f"{task_name}_panthera.yml"
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Cannot derive {task_config}.yml: base config {base_path} does not exist"
        )
    with base_path.open("r", encoding="utf-8") as file:
        args = yaml.load(file.read(), Loader=yaml.FullLoader)

    embodiment = args.get("embodiment")
    if not isinstance(embodiment, list) or not embodiment or embodiment[0] != "panthera-6dof":
        raise ValueError(
            f"Cannot derive Panthera single-arm config from embodiment={embodiment!r}"
        )
    args["arm_mode"] = "single"
    args["embodiment"] = ["panthera-6dof"]
    return args


def start_physics_validation(task_env, args):
    if not args.get("physics_validation", False):
        return None

    from script.validation.physics_monitor import PhysicsMonitor
    from script.validation.rules import create_rule

    monitor = PhysicsMonitor(create_rule(args["task_name"]))
    monitor.start(task_env)
    task_env.add_physics_step_observer(monitor)
    return monitor


def finish_physics_validation(task_env, monitor, *, episode, seed, phase):
    if monitor is None:
        return None
    try:
        report = monitor.finalize(task_env)
    finally:
        task_env.clear_physics_step_observers()
    report.update({
        "episode": episode,
        "seed": seed,
        "phase": phase,
        "plan_success": bool(task_env.plan_success),
    })
    if getattr(task_env, "single_arm_mode", False):
        report["single_arm_setup"] = task_env.single_arm_setup
    report["collection_validation_passed"] = bool(
        report["physical_validation_passed"] and task_env.plan_success
    )
    return report


def write_physics_report(save_path, report, relative_path):
    if report is None:
        return None
    report_path = Path(save_path) / "physics_validation" / relative_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report_path


def prepare_task_and_args(task_name, task_config):
    """Load a task and resolve its complete RoboTwin collection configuration.

    This is shared by the normal collector and the trajectory replay entry
    point so that embodiment and arm-mode handling stay in one place.
    """
    task = class_decorator(task_name)
    args = load_collection_config(task_name, task_config)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment files for {embodiment_type!r}")
        return robot_file

    arm_mode = args.get("arm_mode")
    if arm_mode is None:
        # Preserve the upstream meaning: one embodiment item denotes a
        # combined dual-arm URDF, while three items denote two independent
        # robot instances.
        arm_mode = "combined" if len(embodiment_type) == 1 else "dual"
    if arm_mode not in {"single", "dual", "combined"}:
        raise ValueError(f"Unsupported arm_mode: {arm_mode!r}")
    args["arm_mode"] = arm_mode

    if arm_mode == "single":
        if len(embodiment_type) != 1:
            raise ValueError("arm_mode=single requires embodiment: [robot_name]")
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        # Keep the legacy right-side configuration available as an internal
        # compatibility view. Robot creates only one physical articulation.
        args["right_robot_file"] = args["left_robot_file"]
        args["dual_arm_embodied"] = False
        args["robot_type"] = embodiment_type[0]
    elif len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if arm_mode == "single":
        embodiment_name = f"{embodiment_type[0]}-single"
    elif len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["data_root"] = os.path.abspath(args["save_path"])
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    return task, args


def main(task_name=None, task_config=None, episode_num=None, save_path=None):
    task, args = prepare_task_and_args(task_name, task_config)
    if episode_num is not None:
        if episode_num < 1:
            raise ValueError("episode_num must be at least 1")
        args["episode_num"] = episode_num
    if save_path is not None:
        args["data_root"] = os.path.abspath(save_path)
        args["save_path"] = os.path.join(
            args["data_root"], str(args["task_name"]), args["task_config"]
        )
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                physics_monitor = start_physics_validation(TASK_ENV, args)
                physics_report = None
                try:
                    TASK_ENV.play_once()
                finally:
                    physics_report = finish_physics_validation(
                        TASK_ENV,
                        physics_monitor,
                        episode=suc_num,
                        seed=epid,
                        phase="seed",
                    )
                report_path = write_physics_report(
                    args["save_path"],
                    physics_report,
                    Path("seed_checks") / f"seed{epid}.json",
                )

                physics_passed = (
                    physics_report is None
                    or physics_report["collection_validation_passed"]
                )
                if TASK_ENV.plan_success and TASK_ENV.check_success() and physics_passed:
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    if report_path is not None:
                        print(f"Physical validation: PASS -> {report_path}")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    if report_path is not None:
                        print(f"Physical validation: FAIL -> {report_path}")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            if args["arm_mode"] == "single":
                args["arm_joint_path"] = traj_data["arm_joint_path"]
            else:
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            physics_monitor = start_physics_validation(TASK_ENV, args)
            physics_report = None
            try:
                info = TASK_ENV.normalize_episode_info(TASK_ENV.play_once())
            finally:
                physics_report = finish_physics_validation(
                    TASK_ENV,
                    physics_monitor,
                    episode=episode_idx,
                    seed=seed_list[episode_idx],
                    phase="replay",
                )
            report_path = write_physics_report(
                args["save_path"],
                physics_report,
                f"episode{episode_idx}_physics.json",
            )
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            task_success = bool(TASK_ENV.check_success())
            physics_passed = (
                physics_report is None
                or physics_report["collection_validation_passed"]
            )
            collection_passed = bool(
                TASK_ENV.plan_success and task_success and physics_passed
            )

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            if not collection_passed:
                report_hint = f"; physics report: {report_path}" if report_path else ""
                raise RuntimeError(
                    f"Episode {episode_idx} replay failed collection validation{report_hint}"
                )

            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()

        subprocess.run(
            [
                sys.executable,
                "utils/generate_episode_instructions.py",
                args["task_name"],
                args["task_config"],
                str(args["language_num"]),
                "--data-root",
                args["data_root"],
            ],
            cwd="description",
            check=True,
        )


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "--episode-num",
        type=int,
        default=None,
        help="Override the episode count from the task config",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Override the root directory used to store collected data",
    )
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(
        task_name=task_name,
        task_config=task_config,
        episode_num=parser.episode_num,
        save_path=parser.save_path,
    )
