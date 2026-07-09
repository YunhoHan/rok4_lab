# RoK4 Lab

Current version: `0.1.0`

RoK4 Lab contains lightweight Isaac Lab scripts and RoK4 asset configuration code used to validate the RoK4 whole-body robot model before building reinforcement-learning tasks.

Large robot assets are distributed separately through Dropbox so this Git repository stays small and can be cloned without Git LFS.

## Repository Layout

```text
source/rok4_tasks/         RoK4 Isaac Lab asset configuration package
scripts/                   Standalone model and actuator check scripts
assets/                    Local asset install location, ignored by git
```

## Documentation

Project notes are kept in `docs/` as editable RST/HTML files and generated PDFs:

| Document | Purpose |
| --- | --- |
| `docs/_build/pdf/rok4_flat_task_structure_ko.pdf` | RoK4 flat task file structure, task registration flow, and config relationships. |
| `docs/_build/pdf/rok4_reward_structure_ko.pdf` | RoK4 reward terms, inherited reward settings, and reward function meanings. |

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
| `check_rok4_joint_monkey.py` | `--mode torque_pd` | Sends changing position targets through `IdealPDActuatorCfg`; use this to inspect torque-PD tracking. |
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

RoK4 uses `IdealPDActuatorCfg` in `source/rok4_tasks/rok4_tasks/assets/robots/rok4.py`.

For this explicit actuator, `stiffness` and `damping` are torque-PD gains (`Kp`, `Kd`), not PhysX position-drive stiffness and damping. Isaac Lab computes:

```text
tau = Kp * (q_des - q) + Kd * (qd_des - qd) + tau_ff
```

The resulting torque is clipped by the configured effort limits and sent to PhysX as joint actuation force.

## Flat RL Task

The first RoK4 learning task is a flat-ground, blind velocity-tracking task based on the Isaac Lab/G1-style
manager-based locomotion structure.

Registered task names:

| Task | Purpose |
| --- | --- |
| `RoK4-Isaac-Velocity-Flat-v0` | Train RoK4 flat-ground velocity tracking with `rok4_train.usd`. |
| `RoK4-Isaac-Velocity-Flat-Play-v0` | Play a trained RoK4 flat-ground policy with the visual `rok4_test.usd` asset. |

The actor observation is proprioceptive and history-based. It does not use camera images, terrain height scans, or
base linear velocity:

```text
5-step history of:
  base_ang_vel
  projected_gravity
  velocity_commands
  joint_pos
  joint_vel
  last_action
```

The action space has 13 dimensions:

```text
left leg  : hip yaw, hip roll, hip pitch, knee pitch, ankle pitch, ankle roll
right leg : hip yaw, hip roll, hip pitch, knee pitch, ankle pitch, ankle roll
torso     : torso yaw
```

RoK4 overrides the parent Isaac Lab locomotion timing without modifying Isaac Lab itself:

| Setting | Value |
| --- | --- |
| `sim.dt` | `0.002 s` |
| physics frequency | `500 Hz` |
| `decimation` | `5` |
| policy/action period | `0.010 s` |
| policy/action frequency | `100 Hz` |

The initial PPO baseline uses RoK4-oriented network and observation-normalization settings with G1-style PPO
algorithm parameters. These values are starting points for flat walking, not final tuned parameters:

| Setting | Value |
| --- | --- |
| steps per env | `24` |
| max iterations | `5000` |
| actor/critic hidden dims | `[512, 256, 128]` |
| actor/critic obs normalization | `True` |
| learning rate | `1.0e-3` |
| entropy coef | `0.008` |
| value loss coef | `1.0` |
| desired KL | `0.01` |

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
  --num_envs 512 \
  --max_iterations 5000 \
  --headless
```

Play a checkpoint:

```bash
./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/play.py \
  --task RoK4-Isaac-Velocity-Flat-Play-v0 \
  --num_envs 16 \
  --checkpoint /path/to/model.pt
```
