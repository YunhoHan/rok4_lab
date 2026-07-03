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

The commands below use the local paths from the original development machine as examples. Replace them with your own Isaac Lab and RoK4 Lab checkout paths.

Example paths used in this README:

```text
ISAACLAB_DIR=/home/rclab/IsaacLab
ROK4LAB_DIR=/home/rclab/rok4_lab
```

## Download Assets

Download and extract the RoK4 whole-body asset bundle into this repository:

```bash
export ROK4LAB_DIR=/home/rclab/rok4_lab
cd ${ROK4LAB_DIR}

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
export ISAACLAB_DIR=/home/rclab/IsaacLab
export ROK4LAB_DIR=/home/rclab/rok4_lab

cd ${ISAACLAB_DIR}
conda activate env_isaaclab
```

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
