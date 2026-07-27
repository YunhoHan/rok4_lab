# Changelog

## Unreleased

### Changed

- Changed the training command population from three episode roles to balanced `mixed`, `standing`, `walking`, `x`,
  `y`, `yaw`, and `x_yaw` roles, with asynchronous standing transitions for every role except fixed standing and
  always-walking environments.
- Changed the exact-zero standing pose penalty from squared joint error at `-1.0` to absolute joint error at `-0.2`,
  retaining the role-based standing mask while reducing the foot-dragging observed with the `-1.0` L1 experiment.
- Disabled the world-flat contact-foot orientation reward and reduced the terminal ADAPT actuator-pair gains from
  `Kp=120, Kd=9` to `Kp=80, Kd=7.5` to test passive toe-off and terrain compliance without changing the standing or
  feet-air-time rewards.
- Removed the unused nominal actuator limits and split the actuator limit scaling into independent torque and velocity
  factors.
- Restored the PhysX joint-effort safety limits to the unscaled mechanical maxima while retaining the 90% actuator
  torque limit for control and reward evaluation.
- Changed the first actuator-interface experiment to reuse the previous Isaac Lab PD gains before testing the higher
  Isaac Gym actuator gains.
- Changed the flat policy interface from joint coordinates to 13 actuator coordinates while preserving the 13D action
  and 240D observation tensor shapes.
- Changed torque, velocity, acceleration, limit, and action-smoothness rewards to operate in RoK4 actuator coordinates.
- Restored the torso-yaw deviation penalty to the G1-style `-0.1` and increased the flat-orientation penalty through
  `-2.0` to `-5.0` to isolate whether excessive body roll causes lateral hopping and expose over-constraint effects.
- Relaxed the combined hip-yaw/hip-roll deviation penalty from `-0.1` to `-0.05` so RoK4's rotated hip axes can
  generate lateral foot placement while the stronger root-orientation penalty discourages torso-lean hopping.
- Reduced the actuator-torque penalty from `-1.0e-5` to `-2.0e-6` and changed first/second action-rate penalties from
  scaled target offsets at `-0.1`/`-0.05` to clipped raw actions at `-0.005`/`-0.0005`, retaining mild smoothing
  without making large-range actuator targets disproportionately expensive. Hip-pitch and knee action differences
  retain their `0.5` squared-error multiplier.
- Restored the feet-air-time threshold from `0.55 s` to the G1 Flat value of `0.4 s`.
- Changed the independently sampled exact-zero standing-command environment ratio from 2% to 5%.
- Changed training commands from G1-style world-heading tracking to direct base velocity `[vx, vy, wz]` sampling.
- Changed the experimental reference policy to the symmetry-augmented actuator-space
  `2026-07-24_19-34-26_symmetry_aug_nojumps2_swing_roll100_fresh/model_9999.pt` run, establishing the pre-domain-
  randomization and pre-observation-noise-tuning comparison point.

### Added

- Added a Play/Teleop velocity monitor showing the selected environment's scaled command alongside measured yaw-frame
  linear velocity, world-Z angular velocity, and planar speed.
- Recorded the `Yunho Symmetry ADAPT v1` flat-walking baseline with symmetry augmentation, asynchronous command roles,
  bilateral-flight suppression, signed anti-cross shaping, and swing-foot roll shaping.
- Added a `no_jumps` reward term at weight `-2.0` using Isaac Lab's `desired_contacts` function to penalize intervals
  where neither RoK4 foot has contact, while preserving valid single-foot support and toe-off.
- Added a swing-only foot-roll penalty at weight `-1.0` that suppresses medial/lateral sole tilt without constraining
  swing-foot pitch, yaw, toe-off, or supporting-foot orientation.
- Added an anti-cross reward that preserves signed left-right foot separation in the base yaw frame, penalizing
  lateral widths below `0.16 m` in both standing and moving states without constraining maximum stance width.
- Added paper-aligned RSL-RL left-right symmetry data augmentation for the 240D term-major policy history, 3D
  privileged critic velocity, and 13D ADAPT actuator action while keeping mirror loss disabled.
- Added unit tests for RoK4 mirror indexing, double-mirror recovery, actor/critic batch doubling, and the exact
  ``J P_psi = P_q J`` relationship between ADAPT actuator exchange and joint ankle-roll reflection.
- Added a Play/Teleop `RoK4 Push Test` UI with directional and random world-frame root-velocity disturbances applied
  safely at policy-step boundaries.
- Added a RoK4-local velocity command term that forces every training environment to an exact-zero command for a
  uniformly sampled `1.5-3.0 s` every `10 s`, then resamples all commands for walking-to-standing transition training.
- Added contact-gated foot-flatness and yaw-frame stance-width penalty functions. Both reward terms are currently
  disabled while passive foot compliance and command-mode behavior are evaluated.
- Added an Isaac Lab standing-environment-mask-gated pose penalty over all 13 RoK4 joints to prevent zero-velocity
  stepping without suppressing small non-zero velocity commands.
- Added a keyboard `R` callback that safely clears the teleoperation command, resets the simulated environment under
  inference mode, and resets the policy state. The existing `L` binding continues to clear only the command.
- Added training-video examples for the upstream `--video`, `--video_length`, and `--video_interval` options.
- Added a dedicated Korean ADAPT control-structure document covering matrix transforms, action/actuator relationships,
  target and state origins, velocity-target handling, explicit PD flow, and torque limits.
- Added a configurable RoK4 ADAPT transmission and actuator-space explicit PD model with actuator torque and velocity
  safety limits.
- Added actuator-space action processing and proprioceptive observation terms while retaining the existing observation
  history length and symmetric actor-critic configuration.
- Added env-0 left/right foot total ground-reaction-force arrows and a live force-magnitude panel to the RoK4 contact
  sensor debug visualization.
- Recorded training run `2026-07-19_18-32-43_adapt_raw_action_relaxed_rewards` at checkpoint `model_4999.pt` as the
  experimental `Yunho ADAPT v1` flat-walking baseline.

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
