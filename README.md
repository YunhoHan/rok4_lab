# RoK4 Lab

Current project version: `0.2.0`

Flat walking baseline: `Yunho Directional Touchdown ADAPT v1` (experimental)

Current directional-gait reference policy: run
`2026-08-03_14-58-46_touchdown_air_symmetric_x_fastforward_fresh20k`, checkpoint `model_19999.pt`.
Teleop evaluation confirmed stable forward, backward, lateral, and yaw motion. Checkpoints `model_9999.pt` and
`model_14999.pt` remain useful comparison points, but the TensorBoard tracking metrics are effectively converged by
15k and contact/sliding terms receive only small additional improvements by 20k.

The previous Sim2Sim-validated reference remains run
`2026-07-30_00-50-43_adapt_reset_jointphysics_footdr_fresh`, checkpoint `model_5000.pt`. Preserve both the checkpoint
and its exported ONNX separately; exporting another checkpoint rewrites the run's default `exported/policy.onnx`.

The `yunho/privileged-observation` experiment keeps this policy as its parent baseline and extends only the training
critic. The `yunho/concurrent-state-estimator` ONNX policy retains the 240-value observation input and exposes two
named outputs: the 13-value actuator action and the 3-value body-frame velocity estimate used internally by that
action. Sim2Sim and Sim2Real consumers should read both outputs, apply only `actions` to the actuators, and log
`estimated_base_lin_vel_b` for estimator validation.

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
  │    └─ DelayedPDActuator
  │         └─ RoK4AdaptActuator
  │              ├─ contains RoK4AdaptTransmission
  │              └─ overrides actuator-space compute()
  ├─ IdealPDActuatorCfg
  │    └─ DelayedPDActuatorCfg
  │         └─ RoK4AdaptActuatorCfg
  │              └─ instantiated by rok4.py inside ROK4_TRAIN_CFG
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
  ├─ ActorCritic -> RoK4EstimatorActorCritic
  │    └─ contains 225D -> 3D base-velocity estimator
  ├─ PPO -> RoK4PPO
  │    └─ separate PPO and estimator optimizers
  └─ OnPolicyRunner -> RoK4OnPolicyRunner
       └─ estimator metrics, checkpoint state, and fused export

RoK4 local MDP
  ├─ actions.py       raw actuator action -> psi_target -> q_target
  ├─ commands.py      direct velocity command + periodic standing windows
  ├─ observations.py  joint state -> actuator-space observation
  ├─ rewards.py       actuator-space penalties and action smoothness
  └─ symmetry.py      left-right actor/critic/action batch augmentation

Debug and verification
  ├─ ContactSensor -> RoK4ContactForceVisualizer
  ├─ ManagerBasedRLEnvWindow -> RoK4PushTestWindow
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
naming and centering convention in actuator coordinates. The external Actor/ONNX input remains 240 noisy-history
values. On `yunho/concurrent-state-estimator`, an internal 225D command-free MLP estimates the 3D body-frame base
velocity and concatenates it with the normalized history, so the action MLP consumes 243D without changing the
deployment input. The ONNX graph returns `actions [1,13]` and `estimated_base_lin_vel_b [1,3]` as separate tensors;
these are not concatenated into a 16-value output. During training, the critic receives a separately evaluated clean 240-value history plus ten current privileged values:
`base_lin_vel_b` (3), base height (1), left/right Foot body-origin height (2), contact flags (2), and current air times
(2). This produces a 250-value asymmetric critic input. The ten privileged values are current-frame state, not a
five-frame history.

## Documentation

Project notes are kept in `docs/` as editable RST/HTML files and generated PDFs:

| Document | Purpose |
| --- | --- |
| `docs/_build/pdf/rok4_flat_task_structure_ko.pdf` | RoK4 flat task structure, task registration, DR, and actor/critic/action symmetry augmentation. |
| `docs/_build/pdf/rok4_reward_structure_ko.pdf` | RoK4 reward terms, inherited reward settings, reward/DR separation, and reward function meanings. |
| `docs/_build/pdf/rok4_adapt_control_structure_ko.pdf` | ADAPT matrices, action/actuator object relationships, target/state origins, explicit PD call flow, and torque limits. |
| `docs/_build/pdf/rok4_randomization_and_noise_ko.pdf` | Observation noise, reset randomization, physics DR, sampling cadence, and Train/Play/Teleop differences. |
| `docs/_build/pdf/rok4_concurrent_state_estimator_ko.pdf` | Concurrent 225D-to-3D base-velocity estimator, PPO gradient separation, symmetry, checkpoints, and the fused two-output ONNX contract. |

The development branches intentionally remain independent:

| Branch | Purpose |
| --- | --- |
| `main` | Published symmetric ADAPT walking baseline. |
| `yunho/adapt-actuator-interface` | Historical actuator-interface development line. |
| `yunho/symmetry-augmentation` | Sim2Sim-validated symmetry and DR baseline. |
| `yunho/directional-gait-rework` | Touchdown air-time and directional command-role baseline described here. |
| `yunho/privileged-observation` | Clean critic history, current privileged foot state, and height/clearance reward experiment. |
| `yunho/concurrent-state-estimator` | Concurrent command-free base-velocity estimator while retaining the 240D deployment input and 250D critic. |

Documentation updates on one branch do not imply merging or moving the other branch pointers.

Documentation uses `${ROK4_LAB_ROOT}` for this repository root and `${ISAACLAB_ROOT}` for the Isaac Lab repository
root. Set them to the actual clone locations instead of copying a machine-specific `/home/<user>/...` path:

```bash
export ROK4_LAB_ROOT="${HOME}/rok4_lab"
export ISAACLAB_ROOT="${HOME}/IsaacLab"
```

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

Before ADAPT conversion, the actuator passes `q_target` through the position `DelayBuffer` inherited from Isaac Lab's
`DelayedPDActuator`, with
`min_delay=max_delay=2` physics steps. At the current `2 ms` physics period this reproduces the measured fixed
`4 ms` command-path delay. Current `q` and `q_dot` feedback are not delayed, so this models target communication and
application latency rather than a stale encoder feedback loop. The buffer is cleared on each environment reset and
holds the first available target while its two-sample history fills. This actuator-side state is not part of the
240D Actor observation or the exported ONNX policy.

`DelayedPDActuator` also owns the velocity-target and feed-forward-effort delay buffers and reset logic. The current
position-action task leaves those two command tensors at zero, so their delayed values remain zero. RoK4 overrides
`compute()` instead of calling the parent's ordinary joint-space PD implementation: only buffer ownership and reset
behavior are inherited, while ADAPT conversion, actuator-space PD, actuator torque clipping, and joint-effort mapping
remain RoK4-specific.

The current compliance experiment uses `Kp/Kd=240/12` for hip yaw/roll,
`160/8` for the first ADAPT-coupled pair associated with hip pitch/knee, and `80/8` for the final ankle-side pair.
The torso-yaw gain remains `100/5`. These are diagonal actuator-space gains applied to `psi`, not direct joint-space
gains. The resulting joint-space gain is `K_q=J^-T K_psi J^-1`, so knee and ankle-pitch errors remain coupled while
the sagittal chain and ankle stiffness retain the experimentally stable real-robot range. The per-leg actuator-space
ratios are `20:1, 20:1, 20:1, 20:1, 10:1, 10:1`; ADAPT coupling makes the effective joint-space knee diagonal ratio
`15:1`. The touchdown-force implementation remains available for comparison, but the current reward configuration
sets the term to `None`; GRF is inspected through debug visualization instead of shaping this experiment.

Self-collision is enabled in the RoK4 articulation config through `enabled_self_collisions=True`.

## Symmetry Augmentation

The `yunho/symmetry-augmentation` experiment uses RSL-RL's symmetry data-augmentation path, following the corrected
on-policy PPO formulation from *Symmetry Considerations for Learning Task Symmetric Robot Policies* (Mittal et al.,
2024). Each PPO mini-batch keeps its original samples and appends one left-right mirrored copy. It does not create a
second network or add a weighted mirror loss. On this branch, it preserves the 240D actor and mirrors the separate
240D clean critic history plus the 10D current privileged state, for a total critic input of 250D.

RoK4's mirror callback transforms all five history samples in the term-major Isaac Lab layout, the current privileged
base/foot state, and the 13D raw actuator action. Base height is preserved, lateral base velocity changes sign, and
left/right foot height, contact, and air-time values are swapped. For each ADAPT leg block, exchanging the final two
actuator coordinates preserves joint ankle pitch and reverses joint ankle roll. Unit tests verify this relationship
through `J P_psi = P_q J`, verify that mirroring twice recovers the original sample, and verify the doubled TensorDict
batch.

Start this experiment as a fresh run rather than resuming the pre-symmetry checkpoint. The regular training command
uses augmentation automatically on this branch because `RoK4FlatPPORunnerCfg.algorithm.symmetry_cfg` enables data
augmentation and explicitly disables mirror loss.

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
- a `RoK4 Contact Forces` panel with the left/right force magnitudes in newtons,
- a RoK4-specific two-series plot retaining a fixed 3.0-second, 301-sample history of left/right `|F|`, with a fixed
  `0-4000 N` vertical axis, 0.1-second vertical grid lines, and labeled 0.5-second elapsed physics-time ticks. The
  elapsed time remains monotonic across environment resets and does not use Isaac Sim's looping timeline range.

For each foot, the visualizer adds the ground-filtered world-frame normal force and tangential contact force reported by
PhysX. Arrow direction follows this total `[Fx, Fy, Fz]` vector, while arrow length and the numeric panel use
`sqrt(Fx^2 + Fy^2 + Fz^2)`. The arrow origin is shifted along the force direction so its tail starts just above the foot
instead of clipping into the ground. This is one resultant GRF arrow per foot, not separate arrows for each axis and not
a six-axis ankle force/torque sensor. The graph samples the latest env-0 sensor value at the rendering update rate; it is
not a 500 Hz physics-substep impact trace. The visualizer is disabled by default and does not run in normal headless
training. The panel intentionally exposes no filtering, integration, derivative, autoscale, or editable-limit controls.

### Manual Push Test

The local Play and Teleop launchers add a `RoK4 Velocity Monitor` frame to the standard Isaac Lab window. It displays
the selected environment's final command and measured robot velocity together. Command `vx`, `vy`, `wz`, and planar
speed `|vxy|` are shown after gamepad/keyboard scaling. Measured world linear velocity is rotated into the same
gravity-aligned yaw frame used by the tracking reward, while measured `wz_world` is the root angular velocity about
the world vertical axis. The actual display also includes `vz` and measured planar speed. It refreshes at 20 Hz to
avoid a GPU synchronization on every physics step.

On the concurrent-state-estimator branch, Teleop also draws three planar velocity arrows above the robot. Green is the
body-frame command, blue is the simulator ground-truth `root_lin_vel_b`, and orange is the estimator output actually
fed to the Actor. All three use the same XY direction and length scale; the orange arrow is raised by `0.08 m` so a
close estimate remains visible. The estimator's `vz` is available through the separate ONNX diagnostic output but is
not included in this planar comparison.

The same window also adds a `RoK4 Push Test` frame. It provides
`+X`, `-X`, `+Y`, `-Y`, and `Random XY` buttons plus a configurable `Delta velocity [m/s]` value. A click queues one
base-yaw-frame root linear-velocity change and applies it at the next policy-step boundary. `+X` is the robot's current
forward direction and `+Y` is its current left direction. At application time, the horizontal delta is rotated by the
robot's current yaw into the world frame; base roll and pitch do not tilt the disturbance. This avoids changing
simulation state from inside an asynchronous UI callback.

The push is applied to the environment selected by `Viewer Settings > Environment Index`; Teleop has one environment,
so it naturally targets environment 0. `Random XY` samples independent base-frame x/y changes from
`[-magnitude, magnitude]`. The default magnitude is `0.5 m/s`.

Closing the Isaac Lab panel with its `X` button only hides it. Select `Window > IsaacLab` from the Isaac Sim main menu
to show it again. The local RoK4 window callback then docks it back into the right-side `Property` tab. For an already
running process launched with an older RoK4 checkout, the equivalent temporary Python Console command is
`omni.ui.Workspace.get_window("IsaacLab").visible = True`.

This control reproduces an impulse-like disturbance by changing root velocity. It is not a sustained force in newtons,
does not alter the policy command, and is available only through the local `play.py` and `play_teleop.py` wrappers. It
does not run during training or headless playback. The implementation is entirely inside `rok4_lab`.

The external actor observation is proprioceptive and history-based. It does not directly contain camera images,
terrain height scans, or measured base linear velocity; the concurrent estimator predicts base velocity internally
from the command-free proprioceptive history:

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
`action_rate_l2` and `second_action_rate_l2` use weights `-0.01` and `-0.005`, retaining the previous Gym first-to-second
order ratio. The command-role ratios use the validated `mixed=0.35` and `standing=0.05` split. The current controlled
ablation changes only the exact-zero standing default-pose penalty from `-0.2` to `-0.05`. Action scaling remains
part of actuator-target generation but is not applied by either smoothness reward. Hip-pitch and knee action indices
`[2, 3, 8, 9]` retain the RoK4-specific `0.5` squared-error multiplier.

The reward functions do not clamp actions internally. In the standard RoK4 RSL-RL train/play path,
`clip_actions=1.0` clamps policy output before `ActionManager`, so the effective reward input is still the clipped raw
action. This differs from K1 Rev1, whose runner leaves `clip_actions=None` and therefore evaluates unclipped raw-action
differences.

The torque, velocity, and acceleration penalties now operate in actuator coordinates, matching the previous Isaac Gym
RoK4 basis. All 13 actuators contribute; hip-pitch and knee indices `[2, 3, 8, 9]` use a `0.5` multiplier.
`actuator_torques_l2`, `actuator_vel_l2`, and `actuator_acc_l2` use weights `-2.0e-6`, `-1.0e-4`, and `-1.0e-8`.

> **Checkpoint compatibility:** The actor remains 13 actions and 240 observations, but their semantics changed from
> joint coordinates to actuator coordinates. The critic now receives 250 values after adding a clean history and
> current privileged base/foot state. Do not resume training from a 243D-critic or joint-space checkpoint with this
> configuration. Start a fresh training run. Actor-only inference exports keep the existing 240D interface.
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

The flat training command ranges are `lin_vel_x=(-0.3, 0.85) m/s`, `lin_vel_y=(-0.3, 0.3) m/s`, and
`ang_vel_z=(-0.6, 0.6) rad/s`. This restores the previous Isaac Gym RoK4 backward command limit and supplements the
continuous mixed distribution with dedicated command-axis roles.

Training uses the RoK4-local `RoK4PeriodicFreezeVelocityCommand`. Normal command sampling uses the parent-compatible
fixed `10 s` interval and directly samples base-frame `[lin_vel_x, lin_vel_y, ang_vel_z]`; world-heading control is
disabled so training, Teleop, ROS `cmd_vel`, and the previous Gym task share the same command meaning. At every
episode reset, each environment independently receives one role for that episode:

| Code role | Motion meaning | Ratio |
| --- | --- | ---: |
| `mixed` | unconstrained `vx`, `vy`, and `wz` | 35% |
| `standing` | exact-zero command for the full episode | 5% |
| `walking` | continuously moving mixed command with `norm([vx, vy]) >= 0.10 m/s` | 5% |
| `x` | symmetric low-speed `vx=+/-[0.10, 0.30] m/s` | 20% |
| `fast_forward` | forward-only `vx=[0.30, 0.85] m/s` | 5% |
| `y` | lateral left/right motion with only `vy` active | 10% |
| `yaw` | clockwise/counter-clockwise turning with only `wz` active | 10% |
| `x_yaw` | symmetric low-speed sagittal motion and turning with `vy=0` | 10% |

The signs inside the `x`, `y`, and `yaw` roles are sampled with equal probability. The `x` and `x_yaw` roles limit
`|vx|` to `[0.10, 0.30] m/s`, so `+0.3` and `-0.3 m/s` occupy the same dedicated training range. Dedicated `y` and
`yaw` commands likewise use minimum magnitudes of `0.10 m/s` and `0.10 rad/s`. The two signs of
`vx` and `wz` are sampled independently in `x_yaw`, giving the four forward/backward and clockwise/counter-clockwise
combinations. The separate `fast_forward` role retains the full positive range up to `0.85 m/s`.

The `mixed`, `x`, `fast_forward`, `y`, `yaw`, and `x_yaw` roles use independent random phases and independently sampled
`1.5-3.0 s` standing windows in a `10 s` cycle. Thus 90% of environments train asynchronous moving-to-standing transitions,
while `standing` remains zero and `walking` never freezes. A normal `10 s` command resampling retains the episode
role and samples a new command within that role; the next environment reset samples a new role. Exact-zero commands
set the command term's standing mask and activate `stand_still_joint_deviation_l1` with weight `-0.05`. This weak
13-joint default-pose bias targets stable two-foot standing while leaving more recovery freedom than the validated
`-0.2` baseline. `rel_standing_envs` is disabled to avoid duplicate standing assignment, and the episode-role scheduler
is disabled in Play and Teleop configurations.

### Episode, Freeze, and Push Timers

Training uses a `20 s` episode timeout. With a `0.01 s` policy period, each full episode contains 2,000 policy steps.
The episode counter belongs to each environment: all environments start at counter zero, but an early termination resets
only the affected environment. For example, an environment reset at global simulation time `3 s` times out near global
time `23 s` if it survives the next full episode. Environments without early termination can remain synchronized at
`20 s`, `40 s`, and so on; desynchronization is a natural consequence of per-environment early resets rather than an
explicit random initial episode phase.

The periodic-freeze schedule is a separate per-environment clock. Every episode reset samples a freeze phase uniformly
from `[0, 10) s` and a freeze duration from `[1.5, 3.0] s`. Consequently, a reset environment does not always wait ten
seconds before stopping and may begin inside a freeze window. Eligible command roles repeat this cycle every `10 s`.

The inherited training push is a third independent per-environment timer. On every episode reset it samples the next
push from `[10, 15] s`; the push adds world-frame root `vx` and `vy` in `[-0.5, 0.5] m/s`. A full `20 s` episode normally
contains one push, while an environment that terminates before `10 s` may receive none. Because the push timer and
freeze phase are sampled independently, a push can occur while moving, during exact-zero standing, or around a command
transition. Play and Teleop disable this automatic push event and provide the manual `RoK4 Push Test` UI instead.

The RoK4-local touchdown feet-air-time reward uses `target_air_time=0.50 s` and `weight=2.0`. It reads each foot's
completed `last_air_time` only when exactly one foot reports first contact and pays `T - 0.50` once on that touchdown
step. It returns zero during swing, continued support, simultaneous two-foot touchdown, and planar commands at or below
`0.05 m/s`. Touchdowns shorter than `0.50 s` are penalized and longer ones are rewarded. There is no maximum-reward
air-time cap; velocity tracking, `no_jumps`, and the other gait terms must therefore balance excessively long single
support. Because this reward is event-based, its TensorBoard magnitude is not directly comparable with the previous
dense, squared, or capped touchdown feet-air-time terms.

With the `0.01 s` RoK4 policy interval, the signed event slope is `2.0 * 0.01 = 0.02`. This matches K1's
`1.0 * 0.02 = 0.02` event slope while retaining RoK4's longer `0.50 s` zero crossing, simultaneous-touchdown mask,
and separate `no_jumps` penalty.

The active term uses the stateful `FeetAirTimeTouchdownBiped` class so the same valid touchdown mask owns both the
reward and its physical-unit mean-air-time statistic. The stateless `feet_air_time_touchdown_biped` function remains
available when aggregation is unnecessary. The Reward Manager's standard `Episode_Reward/feet_air_time` scalar is
the weighted reward contribution, not a time measurement.

This branch adds two height terms without enabling a height scanner. `base_height_l2` uses target `0.907 m` and
weight `-1.0`, measuring root world Z relative to each flat environment origin. `feet_clearance` uses weight `+0.2`
and rewards valid swing feet with
`tanh(v_progress / 0.50) * exp(-(h - 0.054)^2 / 0.04^2)`, where `v_progress` is the non-negative swing-foot speed
along the commanded planar direction in the robot yaw frame. Pure-yaw commands retain the original horizontal-speed
magnitude gate because their feet move in opposite directions. The term is zero in standing environments and while
neither foot is in swing. The clearance height is the `Foot_Link` body-origin Z relative to the environment origin, not
a collision-point or ray-scanner measurement. Its `0.054 m` target explicitly combines the desired `0.050 m` sole
clearance with the measured `0.004 m` vertical offset from the sole to the body origin.

The `no_jumps` penalty uses Isaac Lab's `mdp.desired_contacts` with weight `-2.0` and a `1.0 N` force threshold. It
checks the recent contact-force history of both feet and returns a penalty only when neither foot has a qualifying
contact. This preserves normal one-foot support and toe-off while discouraging a true flight phase. It does not
enforce left/right alternation or limit how long one foot may remain the support foot.

The RoK4-local `feet_touchdown_acc` function follows the ROBOTIS K1 event formulation and remains available for
comparison, but its reward term is currently `None`. The `touchdownacc50` experiment used `threshold=50 m/s^2` and
`weight=-0.002`; its episode penalty decreased mainly as touchdown frequency fell, while MuJoCo and hardware landing
impact remained visibly hard. `compute_first_contact(step_dt)` spans the full `10 ms` policy interval, whereas
`body_lin_acc_w` supplies the acceleration at reward evaluation after the final physics substep, so an earlier
`2 ms` contact peak can be missed. A future soft-landing experiment should use a temporally aligned pre-touchdown
vertical-velocity or contact-force-history signal instead.

The current soft-landing baseline uses a stateful, event-only `feet_touchdown_velocity` term with weight `-10.0`.
It stores each Foot body's world-Z velocity from the preceding policy step. At first contact, it applies
`relu(-v_z_prev)^2` only when the previous sample was airborne. This avoids shaping the entire approach or
established stance. `feet_contact_force` is now `None`: GRF remains available in the debug visualization, but it no
longer shapes learning. The earlier Gym-compatible continuous `feet_contact_velocity_l2` function also remains
disabled.

The validated reference is run
`2026-08-12_23-45-39_privileged250_gain240_160_80_air050_w2_tdvel10_ar01_ar2_005_noforce_delay4ms_fresh20k`,
checkpoint `model_19999.pt`. Its final touchdown metric showed a mean pre-touchdown downward speed near `0.044 m/s`
while preserving velocity tracking and increasing mean completed air time. Checkpoints remain under the external
Isaac Lab log directory and are not committed to this repository.

The two active stateful touchdown terms retain GPU episode sums and event counts and publish two event-weighted means
when environments reset: `Metrics/feet_touchdown/mean_pre_touchdown_vertical_speed` [m/s] and
`Metrics/feet_touchdown/mean_air_time` [s]. These are direct physical measurements over detected touchdowns, unlike
`Episode_Reward/*`, and add no per-step GPU-to-CPU logging synchronization. These choices affect training rewards and
diagnostics only; they do not change observations, Actor/Critic dimensions, checkpoints, or ONNX interfaces.

The root `flat_orientation_l2` penalty uses weight `-5.0`. This intentionally large step from the previous `-2.0`
experiment tests whether excessive body roll is the cause of one-foot lateral hopping and also exposes any loss of
natural weight transfer, velocity tracking, or step length. Feet-air-time, actuator gains, and command-role ratios
remain unchanged for an isolated comparison.

The combined hip-yaw/hip-roll deviation penalty is relaxed from `-0.1` to `-0.05`. RoK4's rotated hip joint frames
make lateral foot placement depend on both named axes, so this paired experiment keeps the torso more upright while
allowing the legs to generate lateral steps. The signed anti-cross reward remains active, but it does not impose a
maximum stance width; excessive widening must therefore be checked during playback.

A separate hip-pitch deviation term uses weight `-0.005`. Its weaker, independently logged penalty mildly limits
excessive whole-leg sagittal swing without applying the `-0.05` hip-yaw/hip-roll constraint to the primary fore-aft
gait joint. This is a global default-pose deviation penalty, not a swing-phase knee-flexion target.

The contact-gated `feet_flat_orientation_l2` function remains available for diagnostics, but its reward term is
currently `None`. It measures foot tilt against world up, which is useful for a flat-ground experiment but can oppose
toe-off and terrain-normal alignment. The current experiment instead relies on the reduced ankle-side actuator gains
for passive contact adaptation and a weak `stand_still_joint_deviation_l1=-0.05` term for exact-zero-command posture
stability.

The active `feet_swing_roll_l2` term uses weight `-1.0` to discourage inward or outward sole roll only while a foot
is airborne. The companion `feet_swing_pitch_l2` term applies the same yaw-removed sole-normal calculation to the
forward component with weight `-0.1`. This mild one-tenth pitch penalty discourages persistent toe-up recovery without
forcing the swing sole fully level; foot yaw remains unconstrained, and feet in contact receive no contribution from
either term.

The `feet_lateral_separation_l2` anti-cross reward is active with `minimum_width=0.16 m` and `weight=-2.0`. It rotates
the left-minus-right foot position into the base yaw frame and preserves the lateral sign, so a narrow stance receives
a soft quadratic penalty and a left/right foot swap receives a larger one. It has no maximum-width term and applies in
both standing and moving states; the normal `0.21 m` standing width therefore receives no penalty.

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
Both local launchers expose the same `RoK4 Push Test` UI, so disturbance recovery can be checked with either fixed or
manually controlled velocity commands.

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
  commands.py                     Adds role-based command sampling and asynchronous standing windows
  events.py                       Owns correlated foot-friction and randomized joint-reset events
  observations.py                 Converts joint state to actuator-space observations
  rewards.py                      Owns actuator-space reward calculations and action smoothness terms
```

Changes after baseline commit `d949d40`:

| Area | Baseline delta | Current setting |
| --- | --- | --- |
| Policy observation noise | unchanged | `base_ang_vel +-0.2`, `projected_gravity +-0.05`, `actuator_pos +-0.01 rad`, `actuator_vel +-1.5 rad/s`; no noise on command or last action |
| Physics DR | changed | correlated Foot material DR plus per-environment/joint static friction, viscous friction, and armature scaling |
| Reset state | changed | default-pose scaling remains `0.9-1.1`; joint velocity changed from fixed zero to `U(-0.1,0.1) rad/s` |
| Termination/timeout | unchanged | illegal-contact termination and the `20 s` episode timeout retain their baseline behavior |

Current DR groups:

| Group | Mode | Current setting |
| --- | --- | --- |
| Robot/Foot physics material | `startup` | all robot shapes first use nominal `0.8/0.6`; each environment then overrides both Foot shapes with shared `mu_static=0.5-0.9`, `mu_dynamic=0.75*mu_static` (`0.375-0.675`), restitution `0.1-0.3` |
| Joint physics | `startup` | independently scales each environment/joint's nominal `ROK4_STATIC_FRICTION`, `ROK4_VISCOUS_FRICTION`, and `ROK4_ARMATURE` by `U(0.8,1.2)` |
| Body mass | `startup` | base scale `0.9-1.1`; upper/lower scale `0.9-1.25` |
| Body COM | `startup` | base x/y/z `+-0.01 m`; upper x/y/z `+-0.03 m`; lower x/y/z `+-0.005 m` |
| External base wrench | `reset` | currently zero force/torque |
| Reset joint pose | `reset` | each environment/joint independently scales its default position by `0.9-1.1`; a zero default remains zero |
| Reset joint velocity | `reset` | each environment/joint independently samples `U(-0.1,0.1) rad/s` |
| Reset base pose/velocity | `reset` | mild x/y/yaw pose and velocity perturbation |
| Root XY velocity push | `interval` | per-environment `[10,15] s` timer; adds world-frame x/y velocity in `[-0.5,0.5] m/s` |

The G1/Digit-style nominal robot material is static friction `0.8` and dynamic friction `0.6` for every collision
shape. The Foot override range is active only in the training task. Play and Teleop also fix the Foot shapes at
`0.8/0.6`, so repeated checkpoint comparisons do not change contact friction between launches.

Joint-physics DR uses Isaac Lab's `randomize_joint_parameters` startup event. Static friction, viscous friction, and
armature each receive independent uniform scale samples for every environment and joint. This event changes the PhysX
joint properties initialized from `ROK4_STATIC_FRICTION`, `ROK4_VISCOUS_FRICTION`, and `ROK4_ARMATURE`; it does not
change the actuator-space `ROK4_ACTUATOR_KP` or `ROK4_ACTUATOR_KD`. Play and Teleop remove this event and use the
nominal joint properties.

The initial PPO baseline uses RoK4-oriented network and observation-normalization settings with G1-style PPO
algorithm parameters. These values are starting points for flat walking, not final tuned parameters:

| Setting | Value |
| --- | --- |
| steps per env | `24` |
| max iterations | `5000` |
| actor/critic hidden dims | `[512, 256, 128]` |
| actor/critic obs normalization | `True` |
| actor input | `policy` history, `240` values |
| critic input | clean `critic` history 240D + current `privileged` state 10D, `250` values |
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
  --run_name symmetry_aug_fresh
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
  --run_name symmetry_aug_fresh_video
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

In the Isaac Lab window, expand `RoK4 Push Test`, choose the velocity-change magnitude, and click a direction. For a
multi-environment Play run, first choose the target using `Viewer Settings > Environment Index`. The direction buttons
use the robot's current base-yaw frame, so `+X` remains robot-forward after it turns. If the panel was closed, reopen and
re-dock it with `Window > IsaacLab`.

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
stick command is clamped to `[-1, 1]` and scaled to the training limits: `vx=(-0.3, 0.85) m/s`,
`vy=(-0.3, 0.3) m/s`, and `wz=(-0.6, 0.6) rad/s`. No ROS 2 bridge, `/joy` subscriber, or IPC process is required for
this native Isaac Lab input path.

For estimator checkpoints, the viewport shows green command, blue simulator ground-truth, and orange estimated planar
base-velocity arrows. The orange arrow is diagnostic only and does not change the command, action, or exported policy.

Keyboard input uses the same script and command pipeline:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/play_teleop.py \
  --task RoK4-Isaac-Velocity-Flat-Teleop-v0 \
  --teleop_device keyboard \
  --teleop_keyboard_step 0.05 \
  --checkpoint /path/to/model.pt \
  --real-time
```

Use Up/Down for forward/backward, Left/Right for left/right lateral motion, `Z`/`X` for positive/negative yaw,
`L` to reset only the keyboard command, and `R` to reset the simulated environment and policy state. Click the Isaac
Sim viewport before pressing the keys so it receives keyboard events. Each key press accumulates `0.05 m/s` on a
linear axis or `0.05 rad/s` on yaw by default. Releasing a key preserves the command, and pressing the opposite key
reduces that component by the same increment. Commands are clamped to the policy training ranges. The
`--teleop_keyboard_step` option changes the common numerical increment. The `R` callback queues a one-shot request;
the inference loop then clears the accumulated command, resets the environment under `torch.inference_mode()`, and
resets the policy state. The teleoperation command is written once per 100 Hz policy loop; the policy observes the
new command on the following loop, giving one policy-period (`10 ms`) command latency.
