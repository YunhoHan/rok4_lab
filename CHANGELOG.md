# Changelog

## Unreleased

## 0.2.0 - 2026-07-14

### Added

- Added the first RoK4 flat-ground RSL-RL velocity-tracking task.
- Added Isaac Lab RSL-RL wrapper scripts that register RoK4 tasks without modifying Isaac Lab.
- Added flat task training and playback instructions.
- Added RoK4-local PPO KL logging with `Loss/kl` and `Loss/kl_max` TensorBoard metrics.
- Added a RoK4 teleoperation task and native Isaac Lab gamepad/keyboard playback script.

### Changed

- Changed the RoK4 flat task timing to 500 Hz physics and 100 Hz policy/action updates.
- Changed the RoK4 flat play task to spawn the visual `rok4_test.usd` asset.
- Changed the RoK4 flat PPO entropy coefficient from `0.008` to `0.002` after evaluating the low-noise `0.001`
  baseline.

## 0.1.0 - 2026-07-03

### Added

- Added the initial RoK4 Isaac Lab asset configuration.
- Added zero-command, sinusoidal-command, and joint-limit check scripts.
- Added Dropbox-based asset download instructions for the RoK4 whole-body bundle.
- Added explicit torque-PD actuator notes for RoK4 control checks.
