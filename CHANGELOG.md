# Changelog

## Unreleased

### Changed

- Raised the event-only touchdown-velocity penalty from `-2.0` to `-10.0` after the stronger setting reduced mean
  pre-touchdown downward speed to approximately `0.044 m/s` without degrading velocity tracking or completed air time.
  Recorded the validated `2026-08-12_23-45-39.../model_19999.pt` configuration as the no-GRF-shaping baseline.
- Replaced capped touchdown air-time shaping with the uncapped signed event reward `T - 0.50 s`, increased its weight
  from `0.75` to `2.0` to match K1's `weight * policy_dt = 0.02` touchdown-error slope, and disabled the GRF-based
  `feet_contact_force` reward while preserving GRF visualization.
- Increased first- and second-order raw-action smoothness weights to `-0.01` and `-0.005`. The reward functions do not clamp
  internally; the standard RoK4 runner continues to provide `[-1, 1]` actions through `clip_actions=1.0`.
- Set the per-leg actuator-space gain profile to `[240, 240, 160, 160, 80, 80] / [12, 12, 8, 8, 8, 8]` after the
  lower-stiffness profile produced low-frequency real-robot joint oscillation. This retains `20:1` hip-side and
  `10:1` ankle-side ratios while retaining the measured 4 ms actuator command delay.
- Documented the virtual-work derivation of the effective joint-space gain and the tradeoff between diagonal actuator
  gains and diagonal joint gains.
- Lowered the swing-foot body-origin target from `0.075 m` to `0.054 m`, representing `0.050 m` sole clearance plus
  the measured `0.004 m` offset from the sole to the `Foot_Link` origin.
- Retained touchdown feet-air-time weight `0.75` and touchdown-velocity weight `-2.0`, while increasing the
  touchdown-force weight from `-0.1` to `-0.2` to target impact without recreating the unstable short-step strategy
  observed when all three terms were strengthened together.
- Added a swing-only foot-pitch penalty with weight `-0.1`, one tenth of the swing-roll penalty, to mildly suppress
  persistent toe-up recovery while preserving toe-off and sagittal foot adaptation.
- Changed the actuator-space compliance profile from per-leg
  `[250, 250, 250, 250, 80, 80] / [12.5, 12.5, 12.5, 12.5, 7.5, 7.5]` to
  `[240, 240, 160, 160, 80, 80] / [12, 12, 10, 10, 7.5, 7.5]`, retaining lateral hip
  stiffness while softening the ADAPT-coupled sagittal chain for landing-impact experiments.
- Extended the active touchdown-force landing window from `50 ms` to `100 ms` so the event penalty also observes
  early sole-slap and load-acceptance peaks. Tightened its limit from `1.5` to `1.2` randomized body weight and
  reduced its reward weight from `-0.2` to `-0.1` so it intervenes earlier without over-constraining normal gait.
- Added the measured fixed `4 ms` RoK4 actuator target-command delay by extending Isaac Lab's
  `DelayedPDActuator`/`DelayedPDActuatorCfg`, while retaining the custom ADAPT-space PD and current-state feedback.
- Changed the active touchdown-force penalty from a fixed `1000 N` first-contact sample to the maximum resultant
  force over a finite landing window, using an environment-specific limit derived from randomized total body weight.
- Added event-weighted TensorBoard metrics for mean pre-touchdown vertical speed, peak touchdown normal force, and
  completed foot air time, with each stateful reward term owning its corresponding statistic and no per-step CPU
  synchronization.
- Changed keyboard Teleop from hold-to-command range endpoints to persistent `0.05` physical-unit increments per
  key press, with opposite-key decrement, training-range clamping, and a configurable `--teleop_keyboard_step`.
- Replaced the simultaneous Gym-style approach-velocity and contact-force penalties with one event-only stateful
  touchdown term. It penalizes the squared downward speed from the preceding airborne policy sample when first
  contact occurs without adding custom TensorBoard aggregation.
- Added a ROBOTIS K1-compatible first-contact foot-acceleration function, then disabled its reward term after the
  `touchdownacc50` experiment showed that its policy-step acceleration sample did not reduce observed MuJoCo or
  hardware landing impact. The function and tests remain available for reference.
- Shifted 10% of training environments from `mixed` commands to the balanced `x` role, producing approximately 10%
  pure forward and 10% pure backward episodes, lowered dedicated moving-command minima from `0.15` to `0.10`, and
  exposed a `0.05 m/s` planar-command threshold for touchdown air-time shaping.
- Raised the experimental touchdown interval from `0.50/0.65 s` to `0.65/0.75 s` minimum/maximum-reward air time to
  test a substantially slower gait while retaining the `0.75` reward weight and capped event structure.
- Raised `minimum_air_time` from `0.45 s` to `0.50 s` while retaining `weight=0.75` and
  `maximum_reward_air_time=0.65 s`, strengthening short-step penalties without adding a separate penalty multiplier.
- Renamed the RoK4 touchdown timing parameters to `minimum_air_time` and `maximum_reward_air_time` to distinguish the
  zero-reward reference from the positive-reward cap without changing reward behavior.
- Changed touchdown feet-air-time shaping to the capped signed form `min(T, 0.65) - 0.45`, penalizing short completed
  swings while removing additional incentive beyond `0.65 s`.
- Increased the touchdown feet-air-time weight from `0.5` to `0.75` while retaining its `0.65 s` cap, then changed
  its event shaping from `min(T, 0.65)` to `min(T, 0.65)^2 / 0.65` so short repeated steps earn less reward without
  increasing the maximum touchdown reward.
- Lowered the swing-clearance target from `0.10` to `0.075 m`, narrowed its standard deviation from `0.05` to
  `0.04 m`, and retained the `0.50 m/s` velocity scale to reduce excessive swing-foot height.
- Changed the asymmetric critic from the actor's noisy 240D policy history plus 3D base velocity to a separate clean
  240D proprioceptive history plus 10 current privileged values: base velocity/height and bilateral foot
  height/contact/current-air-time state. The actor and exported policy remain 240D.
- Extended symmetry augmentation to mirror the clean critic history and the 10D privileged state by preserving base
  height, reflecting lateral base velocity, and swapping bilateral foot values.
- Recorded the 20k directional-gait reference run and its 10k/15k/20k TensorBoard comparison, and documented that
  the next critic-only privileged-observation experiment starts fresh on a separate branch without merging the
  existing baseline branches.
- Changed feet-air-time shaping from dense single-stance rewards to a capped one-shot reward when exactly one foot
  touches down, preventing prolonged one-foot support from collecting the saturated value every policy step.
- Split dedicated sagittal commands into symmetric low-speed `vx=+/-[0.15, 0.30] m/s` roles and a separate 5%
  forward-only `vx=[0.30, 0.85] m/s` role while leaving lateral separation constraints unchanged.
- Changed the feet-air-time reward from `weight=0.75, threshold=0.4 s` to `weight=0.5, threshold=0.65 s` to encourage
  slower, longer steps while keeping its maximum pre-`dt` contribution close to the previous value.
- Added a separate `-0.01` hip-pitch deviation penalty to mildly constrain excessive whole-leg swing without coupling
  the primary sagittal gait joints to the stronger hip-yaw/hip-roll deviation penalty.
- Replaced machine-specific `/home/rclab` paths in project documentation with portable `${ROK4_LAB_ROOT}` and
  `${ISAACLAB_ROOT}` repository-root variables.
- Changed robot collision materials to a G1/Digit-style nominal `0.8/0.6` static/dynamic friction baseline, then
  overrode both feet per environment with correlated static friction in `[0.5, 0.9]` and dynamic friction fixed at
  75% of static friction.
- Added startup joint-physics randomization that independently scales each environment and joint's nominal static
  friction, viscous friction, and armature by `U(0.8, 1.2)` while keeping actuator PD gains fixed.
- Changed reset joint velocities from fixed zero to independently sampled `[-0.1, 0.1] rad/s` values while retaining
  the existing default-pose position scaling.
- Clarified the independent per-environment episode, periodic-freeze, and training-push timers in the README and
  generated Korean task/reward documents, including reset behavior and their interaction during disturbances.
- Documented that the post-`d949d40` changes affect physics DR and reset joint velocity while policy observation
  noise and termination/timeout behavior remain unchanged.
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

- Added a two-series live env-0 GRF-magnitude plot to the existing Contact Forces debug panel, retaining 300 rendering
  samples without adding work when sensor debug visualization is disabled.
- Added Gym-compatible soft-landing penalty functions for excessive downward foot velocity near/at touchdown and foot
  normal forces above `1000 N`, using the five-sample physics contact history for impact-force peaks. These legacy
  functions remain available for controlled comparisons but are no longer active reward terms.
- Added a flat-ground base-height penalty with target `0.907 m` and weight `-1.0`.
- Added a standing-masked swing-foot clearance reward with target `0.075 m`, height standard deviation `0.04 m`,
  `tanh(v_xy / 0.50)` speed gating, and weight `0.2`.
- Added a dedicated Korean observation-noise, reset-randomization, and physics-DR document covering sampling cadence,
  randomization granularity, normalization differences, Train/Play/Teleop behavior, and deferred experiments.
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
