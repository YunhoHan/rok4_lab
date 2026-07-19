# Changelog

## Unreleased

### Changed

- Restored the Isaac Gym ankle actuator gains after the reduced-gain experiment did not improve foot-edge contact.
- Removed the unused nominal actuator limits and split the actuator limit scaling into independent torque and velocity
  factors.
- Restored the PhysX joint-effort safety limits to the unscaled mechanical maxima while retaining the 90% actuator
  torque limit for control and reward evaluation.
- Changed the first actuator-interface experiment to reuse the previous Isaac Lab PD gains before testing the higher
  Isaac Gym actuator gains.
- Changed the flat policy interface from joint coordinates to 13 actuator coordinates while preserving the 13D action
  and 240D observation tensor shapes.
- Changed torque, velocity, acceleration, limit, and action-smoothness rewards to operate in RoK4 actuator coordinates.
- Restored the RoK4 flat-orientation and torso-yaw deviation penalty weights to the G1-style `-1.0` and `-0.1`
  values so natural weight transfer is not over-constrained.
- Reduced the actuator-torque penalty from `-1.0e-5` to `-2.0e-6` and changed first/second action-rate penalties from
  scaled target offsets at `-0.1`/`-0.05` to clipped raw actions at `-0.005`/`-0.0005`, retaining mild smoothing
  without making large-range actuator targets disproportionately expensive. Hip-pitch and knee action differences
  retain their `0.5` squared-error multiplier.
- Restored the feet-air-time threshold from `0.55 s` to the G1 Flat value of `0.4 s`.
- Changed the independently sampled exact-zero standing-command environment ratio from 2% to 5%.
- Changed training commands from G1-style world-heading tracking to direct base velocity `[vx, vy, wz]` sampling.
- Increased the standing-pose penalty weight from `-0.5` to `-1.0` after the first standing test retained small
  stepping motions.
- Changed the experimental reference policy from the joint-space Yunho v1 run to the actuator-space
  `2026-07-19_18-32-43_adapt_raw_action_relaxed_rewards` run.

### Added

- Added a RoK4-local velocity command term that forces every training environment to an exact-zero command for a
  uniformly sampled `1.5-3.0 s` every `10 s`, then resamples all commands for walking-to-standing transition training.
- Added a contact-gated foot-flatness penalty and an optional yaw-frame stance-width penalty function. The
  stance-width reward term is currently disabled while its command-mode behavior is evaluated.
- Added an Isaac Lab standing-environment-mask-gated L2 pose penalty over all 13 RoK4 joints to prevent zero-velocity
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
