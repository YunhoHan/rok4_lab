# RoK4 Lab

Current project version: `0.2.0`

Flat walking baseline: `Yunho ADAPT v1` (experimental)

Current actuator-space reference policy: run `2026-07-19_18-32-43_adapt_raw_action_relaxed_rewards`, checkpoint
`model_4999.pt`

Previous joint-space reference policy: run `2026-07-15_17-28-41`, checkpoint `model_4999.pt`

RoK4 Lab contains lightweight Isaac Lab scripts and RoK4 asset configuration code used to validate the RoK4 whole-body robot model before building reinforcement-learning tasks.

Large robot assets are distributed separately through Dropbox so this Git repository stays small and can be cloned without Git LFS.

## Repository Layout

```text
source/rok4_tasks/         RoK4 Isaac Lab asset configuration package
scripts/                   Standalone model and actuator check scripts
assets/                    Local asset install location, ignored by git
docs/                      RST/HTML/PDF project notes
```

Core inheritance and configuration relationships:

```text
Isaac Lab
  ├─ IdealPDActuator
  │    └─ RoK4AdaptActuator
  │         ├─ contains RoK4AdaptTransmission
  │         └─ overrides actuator-space compute()
  ├─ IdealPDActuatorCfg
  │    └─ RoK4AdaptActuatorCfg
  │         └─ instantiated by rok4.py inside ROK4_TRAIN_CFG
  ├─ ActionTerm
  │    └─ RoK4ActuatorPositionAction
  └─ LocomotionVelocityRoughEnvCfg
       └─ RoK4FlatEnvCfg
            ├─ RoK4ActionsCfg
            ├─ RoK4ObservationsCfg
            ├─ RoK4CommandsCfg
            ├─ RoK4RewardsCfg
            └─ ROK4_TRAIN_CFG

RSL-RL
  ├─ PPO -> RoK4PPO
  └─ OnPolicyRunner -> RoK4OnPolicyRunner

RoK4 local MDP
  ├─ actions.py       raw actuator action -> psi_target -> q_target
  ├─ commands.py      direct velocity command + periodic standing windows
  ├─ observations.py  joint state -> actuator-space observation
  └─ rewards.py       actuator-space penalties and action smoothness

Debug and verification
  ├─ ContactSensor -> RoK4ContactForceVisualizer
  ├─ check_rok4_zero.py
  ├─ check_rok4_random.py
  └─ check_rok4_joint_monkey.py
```

`rok4.py` does not inherit from `rok4_adapt.py`. It imports `RoK4AdaptActuatorCfg`, fills it with the concrete RoK4
link lengths, gains, and limits, and stores that config object in `ROK4_TRAIN_CFG`.

`RoK4ObservationsCfg` is a newly defined RoK4 config container rather than a subclass of the parent task's
`ObservationsCfg`. Its nested `PolicyCfg` inherits `ObservationGroupCfg`, while `RoK4FlatEnvCfg.observations` replaces
the inherited observation-config object as a whole. The resulting frame has 48 values and its five-frame flattened
history produces the 240-value policy input. Its actuator state terms use `J^-1 (q - q_default)` position and
`J^-1 (q_dot - q_dot_default)` velocity without a manual observation scale, matching the parent task's relative-state
naming and centering convention in actuator coordinates.

## Documentation

Project notes are kept in `docs/` as editable RST/HTML files and generated PDFs:

| Document | Purpose |
| --- | --- |
| `docs/_build/pdf/rok4_flat_task_structure_ko.pdf` | RoK4 flat task file structure, task registration flow, DR module, and config relationships. |
| `docs/_build/pdf/rok4_reward_structure_ko.pdf` | RoK4 reward terms, inherited reward settings, reward/DR separation, and reward function meanings. |
| `docs/_build/pdf/rok4_adapt_control_structure_ko.pdf` | ADAPT matrices, action/actuator object relationships, target/state origins, explicit PD call flow, and torque limits. |

After downloading the assets, the expected local layout is:

```text
rok4_lab/
  assets/
    rok4_wholebody/
      urdf/
        rok4_train.usd
        rok4_test.usd
      meshes/
      ...
```

## Environment

This project is currently tested with:

- Isaac Lab v2.3.2
- Isaac Sim 5.1.0
- Conda environment: `env_isaaclab`

The commands below assume you already have an Isaac Lab checkout and that this repository was cloned somewhere on your machine. Replace paths only when your local folders are different.

## Download Assets

Download and extract the RoK4 whole-body asset bundle into this repository:

```bash
cd /path/to/your/rok4_lab
ROK4LAB_DIR=$(pwd)

mkdir -p assets
curl -L "https://www.dropbox.com/scl/fi/jkde1dl5qz8m0wso8c8ks/rok4_wholebody.zip?rlkey=v7n4jc9yfe21mi2je1qu0aty2&st=mpw9nl54&dl=1" \
  -o /tmp/rok4_wholebody.zip

unzip /tmp/rok4_wholebody.zip -d assets/
```

Confirm the main USD files exist:

```bash
ls assets/rok4_wholebody/urdf/rok4_train.usd
ls assets/rok4_wholebody/urdf/rok4_test.usd
```

## Quick Checks

Activate Isaac Lab first:

```bash
# Run this from your local RoK4 Lab checkout before moving to Isaac Lab.
cd /path/to/your/rok4_lab
ROK4LAB_DIR=$(pwd)

# Then move to your local Isaac Lab checkout.
cd /path/to/your/IsaacLab
conda activate env_isaaclab
```

### Script Summary

| Script | Purpose | Main options |
| --- | --- | --- |
| `check_rok4_zero.py` | Hold the default pose or run passive zero-effort simulation. | `--asset {train,test}`, `--mode {torque_hold,passive}`, `--fix_root`, `--interactive_drag`, `--disable_gravity`, `--hold_root`, `--root_height`, `--reset_interval` |
| `check_rok4_random.py` | Apply small sinusoidal position targets to all joints through the explicit torque-PD actuator. | `--asset {train,test}`, `--amplitude`, `--frequency`, `--fix_root`, `--disable_gravity`, `--hold_root`, `--root_height`, `--reset_interval` |
| `check_rok4_joint_monkey.py` | Move one joint at a time to inspect joint axes, limits, visuals, and torque-PD tracking. | `--asset {train,test}`, `--joint`, `--mode {teleport,torque_pd}`, `--motion {limits,amplitude}`, `--center`, `--amplitude`, `--frequency`, `--joint_duration`, `--fix_root`, `--disable_gravity`, `--hold_root`, `--root_height`, `--reset_interval` |

Common asset choices:

| Option | Meaning |
| --- | --- |
| `--asset train` | Uses the training-oriented USD asset. |
| `--asset test` | Uses the visual inspection USD asset with mesh visuals. |

Common support options:

| Option | Meaning |
| --- | --- |
| `--fix_root` | Fixes the root link when spawning the articulation. Useful for hanging visual checks. |
| `--disable_gravity` | Disables gravity for all rigid bodies. Useful for inspection without falling. |
| `--hold_root` | Rewrites the root pose and velocity every step. This is stronger than a spawn-time fixed root and is mostly for debugging. |
| `--root_height` | Overrides the initial root height [m]. |
| `--reset_interval` | Number of simulation steps between resets. Use `0` in `check_rok4_joint_monkey.py` to disable periodic resets. |

Mode-specific options:

| Script | Option | Meaning |
| --- | --- | --- |
| `check_rok4_zero.py` | `--mode torque_hold` | Holds `default_joint_pos` with explicit torque PD. |
| `check_rok4_zero.py` | `--mode passive` | Sends zero joint effort commands and does not hold a target pose. |
| `check_rok4_zero.py` | `--interactive_drag` | Forces CPU PhysX so Isaac Sim Shift + left mouse drag can apply link forces without GPU Direct API errors. |
| `check_rok4_joint_monkey.py` | `--mode teleport` | Writes joint state directly. Use this for visual joint-axis and limit inspection. |
| `check_rok4_joint_monkey.py` | `--mode torque_pd` | Sends changing joint targets through the ADAPT actuator-space torque-PD model. |
| `check_rok4_joint_monkey.py` | `--motion limits` | Sweeps each selected joint through its exact joint position limits. |
| `check_rok4_joint_monkey.py` | `--motion amplitude` | Sweeps around `--center` or the default pose by `--amplitude`. |
| `check_rok4_joint_monkey.py` | `--joint all` | Sweeps all RoK4 actuated joints in order. This is the default. |
| `check_rok4_joint_monkey.py` | `--joint JOINT_NAME` | Sweeps only one exact joint name, for example `L_Knee_Pitch_Joint`. |

Zero-command torque hold:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/check_rok4_zero.py \
  --asset test \
  --mode torque_hold \
  --fix_root \
  --root_height 1.2
```

Interactive GUI drag check. This forces CPU PhysX to avoid GPU Direct API errors when using Shift + left mouse drag:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/check_rok4_zero.py \
  --asset test \
  --mode torque_hold \
  --interactive_drag \
  --fix_root \
  --root_height 1.2
```

Joint limit sweep:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/check_rok4_joint_monkey.py \
  --asset test \
  --mode teleport \
  --motion limits \
  --fix_root \
  --root_height 1.2 \
  --joint_duration 5.0
```

Small sinusoidal actuator check:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/check_rok4_random.py \
  --asset test \
  --amplitude 0.05 \
  --frequency 0.5 \
  --fix_root \
  --root_height 1.2
```

## Control Notes

RoK4 uses the local `RoK4AdaptActuatorCfg` explicit actuator in
`source/rok4_tasks/rok4_tasks/assets/robots/rok4_adapt.py`. The configurable link lengths are
`link_alpha=0.09845 m` and `link_beta=0.06 m`.

For each leg's coupled hip-pitch through ankle-roll block, actuator and joint coordinates satisfy:

```text
q = J * psi
psi = inverse(J) * q
tau_psi = Kp * (psi_des - psi) + Kd * (psi_dot_des - psi_dot)
tau_q = inverse(J).T * tau_psi
```

Hip yaw, hip roll, and torso yaw pass through directly. Torque is clipped in actuator coordinates before it is mapped
to joint effort and sent to PhysX. Isaac Lab's PhysX joint drives remain disabled, so a second joint-space PD loop is
not added. The custom actuator also reorders between USD storage order and the canonical policy order
`[left leg, right leg, torso]` internally.

`actions.py` writes `q_target` into Isaac Lab's joint-position-target buffer because that buffer is the input interface
to the explicit actuator model. It is not sent directly to a PhysX position drive. During each simulation step,
`RoK4AdaptActuator.compute()` reads `q_target` and the current joint state, converts them to actuator coordinates,
computes and clips `tau_psi`, maps it to `tau_q`, clears the position target, and returns only joint effort to PhysX.

Self-collision is enabled in the RoK4 articulation config through `enabled_self_collisions=True`.

## Flat RL Task

The first RoK4 learning task is a flat-ground, blind velocity-tracking task based on the Isaac Lab/G1-style
manager-based locomotion structure.

Registered task names:

| Task | Purpose |
| --- | --- |
| `RoK4-Isaac-Velocity-Flat-v0` | Train RoK4 flat-ground velocity tracking with `rok4_train.usd`. |
| `RoK4-Isaac-Velocity-Flat-Play-v0` | Play a trained RoK4 flat-ground policy with the visual `rok4_test.usd` asset. |
| `RoK4-Isaac-Velocity-Flat-Teleop-v0` | Drive the trained policy with an Isaac Lab SE(2) gamepad or keyboard device. |

### Contact-Force Debug View

The RoK4 contact sensor adds a local debug view without modifying Isaac Lab. In the Isaac Sim UI, open
`Scene Debug Visualization` and enable `Contact Forces`. The view displays only environment 0:

- a blue arrow for the left-foot world-frame total ground reaction force,
- a green arrow for the right-foot world-frame total ground reaction force,
- a `RoK4 Contact Forces` panel with the left/right force magnitudes in newtons.

For each foot, the visualizer adds the ground-filtered world-frame normal force and tangential contact force reported by
PhysX. Arrow direction follows this total `[Fx, Fy, Fz]` vector, while arrow length and the numeric panel use
`sqrt(Fx^2 + Fy^2 + Fz^2)`. The arrow origin is shifted along the force direction so its tail starts just above the foot
instead of clipping into the ground. This is one resultant GRF arrow per foot, not separate arrows for each axis and not
a six-axis ankle force/torque sensor. The visualizer is disabled by default and does not run in normal headless training.

The actor observation is proprioceptive and history-based. It does not use camera images, terrain height scans, or
base linear velocity:

```text
5-step history of:
  base_ang_vel
  projected_gravity
  velocity_commands
  actuator_pos relative to default actuator pose
  actuator_vel
  last_action
```

The action space has 13 dimensions:

```text
left leg  : hip yaw, hip roll, hip pitch, knee pitch, ankle pitch, ankle roll
right leg : hip yaw, hip roll, hip pitch, knee pitch, ankle pitch, ankle roll
torso     : torso yaw
```

Action processing follows this pipeline:

```text
policy output
  -> clip raw actuator action to [-1, 1]
  -> actuator_offset = clipped_raw_action * ROK4_ACTUATOR_ACTION_SCALE
  -> psi_default = inverse(J) * q_default
  -> psi_target = psi_default + actuator_offset
  -> q_target = J * psi_target
  -> actuator-space explicit torque-PD
  -> clip tau_psi at 90% actuator torque limits
  -> tau_q = inverse(J).T * tau_psi
  -> PhysX joint effort, guarded by the unscaled mechanical joint-torque limits
```

At environment startup, the action term prints `J`, `J^-1`, `J^T`, and `J^-T`, validates their inverse/transpose
relations and the reference default-pose round trip in CPU FP64 to `1e-5`, and prints a 13-row
`q_default -> psi_default` conversion table. The actual CUDA FP32/TF32 default-pose mapping and round trip are checked
separately to `5e-4` rad so Isaac Lab's training-time TF32 mode does not cause a false matrix-validation failure. This
diagnostic is emitted once for the shared action term rather than once per environment.

The `last_action` observation term stores the clipped raw policy action, not the scaled joint target. When an exported
ONNX/TorchScript policy is called outside Isaac Lab train/play, clamp the policy output to `[-1, 1]` before applying
`ROK4_ACTUATOR_ACTION_SCALE` and before saving it as the next `last_action`.

The action smoothness rewards use clipped raw policy-action differences, matching the G1 first-order convention.
`action_rate_l2` and `second_action_rate_l2` use weights `-0.005` and `-0.0005`; the second-order term is kept at 10% of
the first-order weight so it damps high-frequency action changes without dominating swing motion. Action scaling remains
part of actuator-target generation but is not applied by either smoothness reward. Hip-pitch and knee action indices
`[2, 3, 8, 9]` retain the RoK4-specific `0.5` squared-error multiplier.

The torque, velocity, and acceleration penalties now operate in actuator coordinates, matching the previous Isaac Gym
RoK4 basis. All 13 actuators contribute; hip-pitch and knee indices `[2, 3, 8, 9]` use a `0.5` multiplier.
`actuator_torques_l2`, `actuator_vel_l2`, and `actuator_acc_l2` use weights `-2.0e-6`, `-1.0e-4`, and `-1.0e-8`.

> **Checkpoint compatibility:** The tensor sizes remain 13 actions and 240 observations, but their semantics changed
> from joint coordinates to actuator coordinates. Do not resume or play a joint-space checkpoint, including the
> `2026-07-15_17-28-41` Yunho v1 baseline, with this actuator-interface branch. Start a new training run.
Acceleration remains the physical Isaac Lab acceleration transformed by `inverse(J)`, not Gym's undivided velocity
difference.

`ROK4_ACTUATOR_ACTION_SCALE` matches the previous Isaac Gym RoK4 actuator ranges. In left-leg, right-leg, and torso
order, the values are `[0.4, 0.5, 1.25, 1.5, 0.75, 0.75, 0.4, 0.5, 1.25, 1.5, 0.75, 0.75, 0.4]`. These values are used
for actuator-target calculation; action smoothness is evaluated in clipped raw-action coordinates.

Actuator mechanical torque limits are `[150, 150, 150, 150, 90, 90] N m` per leg and `150 N m` for torso yaw.
Velocity limits are `[12, 12, 12, 12, 15, 15] rad/s` per leg and `12 rad/s` for torso yaw.
`torque_limit_factor=0.9` and `velocity_limit_factor=0.9` independently scale these mechanical maxima. The resulting
torque limit clips the PD command, and the resulting torque/velocity limits are also used by the corresponding limit
rewards. The actuator stores only the mechanical maxima and the two active limits. No separate actuator-position soft
limit is introduced; the existing 95% joint-position soft limits continue to protect actual and target joint positions
after ADAPT mapping.

The PhysX `effort_limit_sim` values remain in joint coordinates and use the unscaled mechanical maxima
`[150, 150, 300, 480, 180, 180] N m` per leg and `150 N m` for torso yaw. They are a final solver safety guard, not a
second 90% control limit. The active control limit is the actuator-space torque limit above.

The Isaac Gym geometry notes map the gait-ready CoM reference `(0.0575 / 2, 0.0, 0.835) m` to the gait-ready base
position `(0.0552, 0.0, 0.907) m`. The straight-leg standing base height is `z=0.919 m`. The articulation root therefore
starts at `(0.0552, 0.0, 0.929) m`, adding `0.010 m` ground clearance above the straight-leg base height. This root
position is separate from `_ROK4_INIT_JOINT_POS`, which matches the active Isaac Gym gait-ready joint pose: hip pitch
`-0.0924 rad`, knee pitch `0.345 rad`, ankle pitch `-0.253 rad`, and zero for hip yaw/roll, ankle roll, and torso yaw.
This joint pose is converted once with `psi_default = inverse(J) * q_default` and becomes the center of actuator actions.

RoK4 overrides the parent Isaac Lab locomotion timing without modifying Isaac Lab itself:

| Setting | Value |
| --- | --- |
| `sim.dt` | `0.002 s` |
| physics frequency | `500 Hz` |
| `decimation` | `5` |
| policy/action period | `0.010 s` |
| policy/action frequency | `100 Hz` |
| contact sensor update period | `0.002 s` |
| contact-force history length | `5` physics samples |

The contact-force history length follows `decimation`, as in Isaac Lab's Digit locomotion configuration. It retains
one contact sample from each 2 ms physics step in a 10 ms policy interval for contact-dependent rewards and
terminations. This sensor buffer is separate from `observations.policy.history_length`, which stacks policy
observations for the actor.

The flat training command ranges are `lin_vel_x=(-0.1, 0.85) m/s`, `lin_vel_y=(-0.3, 0.3) m/s`, and
`ang_vel_z=(-0.6, 0.6) rad/s`. The limited backward range is an intermediate curriculum step before expanding toward
the previous Isaac Gym RoK4 range of `(-0.3, 0.85) m/s`.

Training uses the RoK4-local `RoK4PeriodicFreezeVelocityCommand`. Normal command sampling uses the parent-compatible
fixed `10 s` interval and directly samples base-frame `[lin_vel_x, lin_vel_y, ang_vel_z]`; world-heading control is
disabled so training, Teleop, ROS `cmd_vel`, and the previous Gym task share the same command meaning. Each normal
resampling independently assigns about 5% of environments an exact-zero standing command. Every 10 seconds the
command term then forces all environments to `[0, 0, 0]` for one globally sampled `1.5-3.0 s` window. During that
window it also sets the command term's standing mask, so the `stand_still_joint_deviation_l2` penalty with weight
`-1.0` is active. When the window ends, all environment commands are resampled. This explicitly trains
walking-to-standing and standing-to-walking transitions; the periodic freeze is disabled in Play and Teleop
configurations.

The positive biped feet-air-time reward uses `threshold=0.4 s` and `weight=0.75`, matching the G1 flat baseline. The
threshold caps the rewarded single-stance duration; it is not an exact gait-period target. Its maximum pre-`dt`
contribution is `0.4 * 0.75 = 0.30`.

The `dof_pos_limits` reward applies to all 13 joints in `ROK4_JOINT_ORDER`. It uses each joint's
`soft_joint_pos_limits`, derived from the USD hard limits with `soft_joint_pos_limit_factor=0.95`, and penalizes only
the amount outside those soft limits with weight `-1.0`.

The `joint_action_target_pos_limits` reward is the Isaac Lab counterpart of the previous Isaac Gym RoK4
`penalty_action_limits`. It compares the `actuator_pos` action term's processed joint target
(`J * (psi_default + clipped_raw_action * ROK4_ACTUATOR_ACTION_SCALE)`) against the same 95% soft limits for all 13 joints.
Only target overshoot is summed, with the previous Gym coefficient `-0.001`; `dof_pos_limits` independently checks the
actual simulated joint positions. Both Gym and the current Lab task therefore evaluate the mapped joint target.

The Play task currently uses an exact-zero velocity command to check whether the policy can stand still. The separate
Teleop task accepts a manual base-frame `[lin_vel_x, lin_vel_y, ang_vel_z]` command, disables heading control and
automatic command resampling, uses one visual test-asset environment, and extends the episode timeout to 600 seconds.

RoK4 also owns its termination configuration without modifying Isaac Lab. `RoK4TerminationsCfg` inherits the parent
`time_out`, disables the parent `base_contact`, and adds `illegal_body_contact`. Contact force above `1.0 N` on any
non-foot body terminates the episode; only `L_Foot_Link` and `R_Foot_Link` are excluded. A non-timeout termination
activates `termination_penalty`, while a normal timeout does not. Because the contact sensor reports net contact force,
contact caused by enabled self-collision can also activate this term.

Domain randomization is managed separately from the main flat environment config:

```text
source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/
  flat_env_cfg.py                 Calls apply_rok4_domain_randomization(self)
  domain_randomization_cfg.py     Owns RoK4 DR ranges and event modes
source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/mdp/
  __init__.py                     Re-exports Isaac Lab locomotion mdp plus RoK4 local mdp
  actions.py                      Converts raw actuator actions to mapped joint targets
  commands.py                     Adds periodic all-environment exact-zero command windows
  observations.py                 Converts joint state to actuator-space observations
  rewards.py                      Owns actuator-space reward calculations and action smoothness terms
```

Current DR groups:

| Group | Mode | Current setting |
| --- | --- | --- |
| Foot physics material | `startup` | static friction `0.8`, dynamic friction `0.6`, restitution `0.1-0.3` |
| Body mass | `startup` | base scale `0.9-1.1`; upper/lower scale `0.9-1.25` |
| Body COM | `startup` | base x/y/z `+-0.01 m`; upper x/y/z `+-0.03 m`; lower x/y/z `+-0.005 m` |
| External base wrench | `reset` | currently zero force/torque |
| Reset joint pose | `reset` | default joint positions scaled by `0.9-1.1` |
| Reset base pose/velocity | `reset` | mild x/y/yaw pose and velocity perturbation |

The initial PPO baseline uses RoK4-oriented network and observation-normalization settings with G1-style PPO
algorithm parameters. These values are starting points for flat walking, not final tuned parameters:

| Setting | Value |
| --- | --- |
| steps per env | `24` |
| max iterations | `5000` |
| actor/critic hidden dims | `[512, 256, 128]` |
| actor/critic obs normalization | `True` |
| action clipping | `clip_actions = 1.0` |
| learning rate | `1.0e-3` |
| entropy coef | `0.002` |
| value loss coef | `1.0` |
| desired KL | `0.01` |

RoK4 training uses `scripts/rsl_rl/rok4_ppo.py` to add KL-divergence logging without modifying Isaac Lab or the
installed `rsl_rl` package. The PPO update itself is unchanged except for collecting the KL values that the adaptive
learning-rate schedule already computes. TensorBoard receives:

| TensorBoard tag | Meaning |
| --- | --- |
| `Loss/kl` | Mean KL divergence across all PPO mini-batch updates in one iteration. With 5 epochs and 4 mini-batches, this averages 20 values. |
| `Loss/kl_max` | Maximum KL divergence among those mini-batch updates. This exposes brief update spikes that an iteration mean can hide. |
| `Loss/learning_rate` | Learning rate after adaptive KL scheduling. |

With `desired_kl=0.01`, adaptive scheduling divides the learning rate by `1.5` when a mini-batch KL exceeds `0.02`,
multiplies it by `1.5` when KL is between `0` and `0.005`, and otherwise keeps it unchanged. The lower
`entropy_coef=0.002` is an intermediate setting between the original `0.008` and the low-noise `0.001` experiment. It
keeps substantially less pressure to increase exploration standard deviation than the original setting while avoiding
the strongest exploration collapse observed with `0.001`.
The KL tags are produced only by new training runs started through the RoK4 `scripts/rsl_rl/train.py` wrapper; existing
event files are not modified retroactively.

Deployment reminder:

```text
Isaac Lab train/play.py:
  RslRlVecEnvWrapper applies clip_actions automatically.

External sim2sim/sim2real policy call:
  obs -> saved observation normalizer if it is not embedded in the exported policy
  policy output -> clamp [-1, 1]
  clipped action -> psi_target = psi_default + action * ROK4_ACTUATOR_ACTION_SCALE
  psi_target -> q_target = J * psi_target
```

Train a short smoke test:

```bash
cd /path/to/your/rok4_lab
ROK4LAB_DIR=$(pwd)

cd /path/to/your/IsaacLab
conda activate env_isaaclab

./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/train.py \
  --task RoK4-Isaac-Velocity-Flat-v0 \
  --num_envs 2 \
  --max_iterations 1 \
  --headless
```

Train normally:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/train.py \
  --task RoK4-Isaac-Velocity-Flat-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless \
  --run_name adapt_raw_action_relaxed_rewards
```

Record periodic training videos by adding `--video`. The interval and length count policy/environment steps rather
than PPO iterations. At the current 100 Hz policy rate, the following records a 5-second clip every 100 simulated
seconds and writes it under the run directory's `videos/train/` folder:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/train.py \
  --task RoK4-Isaac-Velocity-Flat-v0 \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless \
  --video \
  --video_length 500 \
  --video_interval 10000 \
  --run_name adapt_raw_action_relaxed_rewards_video
```

Headless recording uses the configured fixed camera, so it cannot be interactively zoomed or rotated while training.
Video rendering adds GPU, runtime, and storage overhead.

Play a checkpoint:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/play.py \
  --task RoK4-Isaac-Velocity-Flat-Play-v0 \
  --num_envs 16 \
  --checkpoint /path/to/model.pt
```

Teleoperate a checkpoint with a connected gamepad, including a DualShock 4 detected by Isaac Sim:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/play_teleop.py \
  --task RoK4-Isaac-Velocity-Flat-Teleop-v0 \
  --teleop_device gamepad \
  --teleop_dead_zone 0.05 \
  --checkpoint /path/to/model.pt \
  --real-time
```

The left stick controls forward/backward and lateral velocity, and the right stick controls yaw velocity. Moving either
stick to the right produces negative lateral/yaw commands, so the robot moves or turns to its right. The normalized
stick command is clamped to `[-1, 1]` and scaled to the training limits: `vx=(-0.1, 0.85) m/s`,
`vy=(-0.3, 0.3) m/s`, and `wz=(-0.6, 0.6) rad/s`. No ROS 2 bridge, `/joy` subscriber, or IPC process is required for
this native Isaac Lab input path.

Keyboard input uses the same script and command pipeline:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/play_teleop.py \
  --task RoK4-Isaac-Velocity-Flat-Teleop-v0 \
  --teleop_device keyboard \
  --checkpoint /path/to/model.pt \
  --real-time
```

Use Up/Down for forward/backward, Left/Right for left/right lateral motion, `Z`/`X` for positive/negative yaw,
`L` to reset only the keyboard command, and `R` to reset the simulated environment and policy state. Click the Isaac
Sim viewport before pressing the keys so it receives keyboard events. Each motion key commands the corresponding
training-range endpoint while held; releasing it removes that component. The `R` callback queues a one-shot request;
the inference loop then clears the device command, resets the environment under `torch.inference_mode()`, and resets
the policy state. The teleoperation command is written once per 100 Hz policy loop; the policy observes the new
command on the following loop, giving one policy-period (`10 ms`) command latency.
