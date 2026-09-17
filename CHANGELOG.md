# Changelog

## Unreleased

### Fixed

- Fixed the flat-task PDF print style to preserve code-block line breaks while wrapping long lines, keeping
  the actuator gain arrays and framework trees readable.
- Fixed toe/heel/lowest-corner velocity diagnostics on 2026-09-09 to combine link-origin velocity with
  link-origin-relative sole offsets. Previously the diagnostic mixed foot COM velocity with those offsets.
  Initially kept the touchdown reward on its existing COM velocity signal and left all reward weights and force metrics
  unchanged. Re-evaluate existing checkpoints for corrected point-speed metrics; old aggregated point speeds
  cannot be directly compared or repaired without the missing per-sample motion data.

### Added

- Added `FeetTouchdownPitchL2` at weight `-1.0` on 2026-09-16 while retaining swing pitch `-0.1`.
  Evaluated the larger previous/current yaw-removed sole-normal X-component error once per observed
  airborne-to-contact transition, summed feet, and excluded reset contacts and continued stance.
  Added regression coverage for sample timing, reset isolation, body ordering, mirroring and yaw invariance.
  Kept the separate 100 ms edge-velocity window, COM velocity, other rewards, gains and deployment unchanged.
  Updated README/RST/HTML/PDF with the event formula, sampling limitations and untrained experiment status;
  withdrew the earlier diagnostic-only proposal without changing committed baseline references.
- Added the 2026-09-14 landing-edge experiment: `FeetTouchdownEdgeVelocityL2` at weight `-0.1` supplements the
  unchanged COM touchdown cost with the fastest descending sole corner. Added the previous airborne sample once,
  followed by current policy samples for a fixed 100 ms per-foot window; contact chatter cannot extend the window.
  Added reset/bounce/timing/rotation/mirroring tests and completed-window early/late edge-speed metrics in m/s.
  Shared the corrected link-origin point-kinematics helper with diagnostics without changing existing metrics.
  Kept GRF shaping disabled, standing/clearance/air-time/gains/DR unchanged, and introduced no recovery gate.
  Documented the completed gain-only comparison separately from this untrained experiment and retained all
  committed baseline references and deployment interfaces.
- Added the 2026-09-08 `feet_standing_contact` experiment at weight `-0.1`: counted each missing foot contact
  only when all three velocity commands were exactly zero. Kept touchdown air-time, estimator inputs, base weights,
  gains, command roles, freeze, and pushes unchanged, with the standing-pose term still disabled. This introduces
  no recovery gate or load-sharing target; zero-command protective steps also incur the new cost.
- Documented the preceding hip-deviation Teleop result and same-checkpoint estimated/true-velocity diagnostic
  separately from unperformed Sim2Sim/Sim2Real validation, without changing any committed baseline reference.

### Changed

- Recorded the September 17 `edgevel01_pre5_window100_tdpitch1_fresh25k` run and `model_24999.pt` as the current
  development baseline on 2026-09-18, using post-training code snapshot `903318c`. Recorded the user's softer-landing
  Isaac Sim keyboard Teleop observation separately from pending MuJoCo Sim2Sim and hardware Sim2Real validation.
  Preserved previous baselines, added checkpoint identity and final-500-iteration comparisons, and synchronized
  README plus reward/task/ADAPT RST, HTML and PDF documents. Corrected stale estimator and gain-validation descriptions.
- Increased only the pre-touchdown part of the sole-edge speed cost by setting `pre_touchdown_scale=5.0`
  on 2026-09-17. Retained the default `1.0` for existing callers, weight `-0.1`, and the 100 ms post-contact
  window. Applied the multiplier after squaring; retained unscaled physical metrics and bounce/reset behavior.
  Added regression tests for pre-only scaling, invalid scales, unchanged window costs, and physical metrics.
  Recorded the completed touchdown-pitch comparison and user-reported stable standing/heel-first Teleop
  separately from the untrained experiment and unreported Sim2Sim/Sim2Real validation. Updated README/RST/HTML/PDF
  without changing gains, other rewards, deployment interfaces, or committed baseline references.
- Restored only `feet_swing_pitch_l2.weight` from `-0.2` to `-0.1` on 2026-09-16 after the completed trial showed
  user-reported worse toe-down swing and a new tilted-foot standing behavior. Kept all other training settings
  unchanged and added no toe-contact penalty, recovery gate, or diagnostic instrumentation. Documented that
  swing-masked average cost is not touchdown posture and that replay requires the matching earlier checkpoint.
- Increased only `feet_swing_pitch_l2.weight` from `-0.1` to `-0.2` on 2026-09-15 for a fresh-training comparison.
  Retained the existing swing-only sole-normal formula, roll `-1.0`, COM touchdown `-10.0`, edge velocity
  `-0.1 / 100 ms`, and metric-only GRF. Kept gains, all other rewards, DR, commands/pushes, and deployment unchanged.
  Recorded the completed edge experiment's log comparison and user-reported airborne heel-down/toe-down rotation
  separately from the untrained follow-up and unreported Sim2Sim/Sim2Real validation; preserved committed baselines.
- Changed the nominal per-leg actuator gains on 2026-09-10 from
  `[240, 240, 160, 160, 80, 80] / [12, 12, 8, 8, 8, 8]` to
  `[240, 240, 180, 180, 120, 120] / [12, 12, 9, 9, 10, 10]` for a fresh comparison.
  Kept torso `100/5`, COM touchdown rewards, all reward weights, DR, observations, commands, pushes, and the
  fixed 4 ms delay unchanged. Gain randomization remains disabled. Updated effective joint gain matrices
  and stale current-gain tables; documented Kd choices as experimental heuristics, not validated optima.
  Recorded completion and user-reported Isaac Sim observations of the preceding COM-restoration run separately
  from pending validation of the new gains. Old checkpoints require their original gains for identical-controller replay.
- Restored `FeetTouchdownVelocityL2` to foot COM velocity on 2026-09-09 after the link-origin trial showed severe
  early termination. Kept the previous policy-step sample, event masks, XY/downward-Z penalty, weight `-10.0`,
  all other training settings, and the corrected toe/heel point diagnostics. The two touchdown-speed metrics
  again describe COM velocity; the `tdlink` run still describes origin velocity under those same tags.
  Updated the regression test to distinguish COM, origin, and post-impact samples. Documented the pending
  fresh COM-restoration run without changing committed baselines or claiming new simulator/hardware validation.
- Changed `FeetTouchdownVelocityL2` from foot COM to foot link-origin world-frame velocity for a separate fresh
  experiment on 2026-09-09. Retained the previous policy-step sample, event masks, XY/downward-Z penalty, weight
  `-10.0`, standing-contact weight `-0.2`, and all other training settings. Existing touchdown-speed metric names
  referred to the link origin during that trial; do not interpret COM curves as the same measurement. Kept the diagnostic
  point-velocity fix and Actor/ONNX interfaces unchanged. Recorded user-reported quiet standing and walking in
  Isaac Sim for `2026-09-08_16-59-33_concurrent_estimator_hip020_standcontact020_fresh25k/model_24999.pt` as the
  COM comparison policy, not as new Sim2Sim/Sim2Real validation or a replacement committed baseline.
- Increased only `feet_standing_contact` from `-0.1` to `-0.2` for a fresh-training follow-up. The `standcontact010`
  run still stepped at zero command in model-5000 Teleop, and its 500-iteration-average contact cost remained near
  `-0.023` through iteration 7122. Kept the reward function, all other weights and environment settings unchanged;
  stronger contact shaping is experimental and may restrict zero-command recovery steps.
- Increased only hip yaw/roll deviation from `-0.1` to `-0.2` for the 2026-09-07 follow-up to the no-standing-pose
  experiment. Kept the standing-pose term disabled and all other training settings unchanged; this targets excessive
  outward leg posture without claiming quiet standing or recovery validation.
- Disabled the command-gated full-body standing pose term for a controlled ablation and strengthened the generic
  base-height, hip-deviation, and first/second raw-action smoothness weights to `-5.0`, `-0.1`/`-0.01`, and
  `-0.05`/`-0.01`, respectively. This tests a K1-inspired but RoK4 100 Hz-specific standing formulation while
  preserving the validated `stand_still_joint_deviation_l1=-0.1` rollback reference.
- Recorded `2026-09-02_15-48-05_concurrent_estimator_tdmetrics_minwidth0165_fresh25k/model_24999.pt`, trained from
  code commit `ce08e9c`, as the current development baseline while preserving the previous mixed-push reference and
  marking Isaac Sim Teleop, MuJoCo Sim2Sim, and hardware Sim2Real validation as pending.
- Increased the experimental signed lateral foot-separation threshold from `0.160 m` to `0.165 m` to provide
  additional Sim2Real clearance between the left and right ankle assemblies without returning to the previously
  aggressive `0.170 m` setting.
- Increased the current experimental exact-zero standing default-pose weight from the validated `-0.05` baseline to
  `-0.1` as a compromise between standing posture retention and disturbance-recovery freedom.
- Added a reward-neutral touchdown diagnostic that reconstructs the four RoK4 sole corners, records toe, heel, and
  lowest-edge pre-touchdown velocity including angular ``omega x r`` motion, and splits world-Z normal-force peaks
  into 0-20 ms impact and 20-100 ms weight-acceptance windows. Teleop now prints these metrics after automatic or
  manual episode resets so existing checkpoints can be diagnosed without retraining.
- Changed pure-yaw swing-clearance shaping from an unsigned planar-speed gate to a 50/50 blend of target-height
  shaping and commanded yaw-tangential progress. This rewards lifting the foot during in-place turns without
  rewarding arbitrary forward/backward swing, while linear-command and standing behavior remain unchanged.
- Extended the event-only touchdown-velocity penalty from downward world-Z speed to the sum of world-XY speed and
  downward world-Z speed squared. Added separate physical-unit TensorBoard metrics for mean pre-touchdown planar and
  vertical speed while retaining foot-wise summation and the existing first-contact event mask.
- Replaced the stale top-level directional-policy label with a validation-specific baseline table that ties each run
  and checkpoint to its matching code commit and distinguishes current development, MuJoCo Sim2Sim, hardware
  Sim2Real, and previous references.
- Added a persistent repository checklist requiring RST/HTML/PDF/changelog synchronization and explicit approval
  before another branch is merged, fast-forwarded, or moved during a baseline update.
- Replaced the automatic world-frame velocity-only push with an environment-local mixed disturbance. Each event uses
  the robot's base-yaw frame and exclusively selects either an instantaneous velocity change or an impulse-equivalent
  finite-duration force pulse, while retaining independent 10-15 s push timers.
- Restored the exact-zero standing full-body default-pose penalty to the validated mixed-push value `-0.05`. A
  controlled `-0.01` ablation produced standing stepping, larger action and torque costs, and higher peak contact
  force; `-0.05` retains stable two-foot standing while preserving disturbance-recovery steps.
- Added `Window > IsaacLab` recovery for the local environment panel, re-docked the panel into the right-side
  `Property` tab whenever it is shown, and changed manual push buttons from world X/Y to the robot's current
  base-yaw frame while retaining a horizontal world-frame velocity update at the policy-step boundary.
- Added a Teleop-only orange planar velocity arrow for the Actor's body-frame estimator output, using the same
  direction and length convention as the existing green command and blue simulator-velocity arrows.
- Exposed the Actor's internal 3D body-frame base-velocity estimate as the separate ONNX output
  `estimated_base_lin_vel_b`, alongside the unchanged 13D `actions` output. The deployment input remains 240D,
  TorchScript remains action-only, and existing estimator checkpoints require only re-export rather than retraining.
- Kept touchdown-force reward shaping disabled while adding a logging-only force term that records the 100 ms landing
  peak in TensorBoard and returns an identically zero reward.
- Raised the event-only touchdown-velocity penalty from `-2.0` to `-10.0` after the stronger setting reduced mean
  pre-touchdown downward speed to approximately `0.044 m/s` without degrading velocity tracking or completed air time.
  Recorded the validated `2026-08-12_23-45-39.../model_19999.pt` configuration as the no-GRF-shaping baseline.
- Replaced capped touchdown air-time shaping with the uncapped signed event reward `T - 0.50 s`, increased its weight
  from `0.75` to `2.0` to match K1's `weight * policy_dt = 0.02` touchdown-error slope, and disabled the GRF-based
  `feet_contact_force` reward while preserving GRF visualization.
- Retained first- and second-order raw-action smoothness weights at `-0.01` and `-0.005` and restored the validated
  `mixed=35%`, full-episode `standing=5%` command-role split. This keeps the weak standing-pose weight as the only
  experimental reward change. The standard RoK4 runner continues to provide `[-1, 1]` actions through
  `clip_actions=1.0`.
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

- Added a concurrently supervised command-free base-velocity estimator with a `225 -> 256 -> 128 -> 3` MLP,
  physical-unit RMSE logging, symmetry-augmented targets, independent optimizer/checkpoint state, and a fused
  `240 -> 13` JIT/ONNX deployment interface.
- Added a Korean estimator architecture document covering observation layouts, MSE/RMSE equations, PPO gradient
  separation, symmetry, checkpoint compatibility, export behavior, and the deferred recovery-gate experiment.
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
