# RoboTwin Policy Physics Validation

This package validates trained policy rollouts inside their original RoboTwin
task scenes. It does not import hardware SDKs or replace SAPIEN physics.

Enable it from `script/eval_policy.py` with the `physics_validation` override.
Reports are written under that evaluation run's `physics_validation/` directory.

Example for the Panthera ACT checkpoint:

```bash
python script/eval_policy.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name move_pillbottle_pad \
  --task_config move_pillbottle_pad_panthera \
  --ckpt_setting act20-single-arm \
  --ckpt_dir policy/ACT/act_ckpt/act-move_pillbottle_pad/move_pillbottle_pad_panthera-single-arm-10-aligned \
  --test_num 1 \
  --physics_validation True
```

The current task rule records both raw evidence and provisional pass/fail
criteria:

- bilateral finger contact and contact duration;
- object lift and table contact while lifted;
- object-to-end-effector relative drift;
- robot penetration, contact impulse, and unexpected environment contacts;
- release support, final speed, and the task's `check_success()` result.

The observer runs after every RoboTwin physics step during policy execution. It
does not use Panthera Host or sampled HDF5 timestamps as a physics source.
Thresholds are marked `provisional` until they have been compared with
deliberately bad rollouts and additional tasks.

The validation package is for policy rollout testing. It intentionally does
not provide a second `_traj_data` replay entry point; collected trajectories
are converted and trained through the ACT tools under `policy/ACT/`.
