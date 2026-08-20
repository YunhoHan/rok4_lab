RoK4 Reward Structure
=============================================================

작성일: 2026-07-15
최종 업데이트: 2026-08-12

.. raw:: html

   <style>
   @media print {
     @page {
       size: A4 landscape;
       margin: 10mm;
     }
     body, main {
       background-color: white;
     }
     table {
       width: 100% !important;
       table-layout: fixed;
       font-size: 8px;
     }
     table td, table th {
       overflow-wrap: anywhere;
       word-break: break-word;
     }
     table .literal, table .pre {
       white-space: normal;
       overflow-wrap: anywhere;
     }
     pre.code {
       break-inside: avoid;
       page-break-inside: avoid;
     }
   }
   </style>

이 문서는 ``RoK4-Isaac-Velocity-Flat-v0`` task의 현재 reward 구조와 reward function 설정을 정리한다.
현재 reward는 Isaac Lab G1 velocity task 구조를 출발점으로 RoK4 ADAPT actuator 좌표, direct velocity
command, standing transition에 맞게 조정한 flat walking baseline이다. 현재 reference는
``2026-07-24_19-34-26_symmetry_aug_nojumps2_swing_roll100_fresh/model_9999.pt`` 이며 experimental
``Yunho Symmetry ADAPT v1`` 으로 기록한다. 이 checkpoint는 domain randomization과 policy observation
noise를 조정하기 전의 비교 기준이며, 이후 실험에서도 보존한다.

아래 경로에서 ``RoK4:`` 는 ``${ROK4_LAB_ROOT}`` 를, ``Isaac Lab:`` 은
``${ISAACLAB_ROOT}`` 를 기준으로 한 상대경로를 뜻한다. 두 변수는 각 사용자가 clone한 저장소 root를 가리킨다.

.. code-block:: bash

   export ROK4_LAB_ROOT="${HOME}/rok4_lab"
   export ISAACLAB_ROOT="${HOME}/IsaacLab"

Observation noise, reset state randomization, physics DR의 상세 범위와 적용 주기는
``docs/rok4_randomization_and_noise_ko.rst`` 를 기준 문서로 사용한다.

관련 파일
-------------------------------------------------

.. list-table::
   :header-rows: 1

   * - 역할
     - 파일
   * - RoK4 flat task 설정
     - RoK4: ``.../config/rok4/flat_env_cfg.py``
   * - RoK4 ADAPT transmission/actuator
     - RoK4: ``.../assets/robots/rok4_adapt.py``
   * - RoK4 actuator action/observation
     - RoK4: ``.../velocity/mdp/actions.py``, ``observations.py``
   * - RoK4 actuator reward 함수
     - RoK4: ``.../velocity/mdp/rewards.py``
   * - RoK4 domain randomization 설정
     - RoK4: ``.../config/rok4/domain_randomization_cfg.py``
   * - 공통 locomotion reward term 기본값
     - Isaac Lab: ``.../velocity/velocity_env_cfg.py``
   * - locomotion 전용 reward 함수
     - Isaac Lab: ``.../velocity/mdp/rewards.py``
   * - Isaac Lab 공통 reward 함수
     - Isaac Lab: ``isaaclab/envs/mdp/rewards.py``
   * - mdp namespace 연결
     - Isaac Lab: ``.../velocity/mdp/__init__.py``

Actuator interface와 reward 연결
-------------------------------------------------

현재 reward는 단순히 joint 값을 actuator라고 이름만 바꾼 것이 아니다. action target, 실제 상태, PD torque가
각각 ADAPT 행렬을 통과하며, reward는 그 runtime actuator 값 또는 mapped joint target을 명시적으로 사용한다.

.. code-block:: text

   rok4.py
     └─ 링크 길이, actuator gain/action scale/limit 설정
          -> rok4_adapt.py
               ├─ RoK4AdaptTransmission: J/J^-1/J^T/J^-T
               └─ RoK4AdaptActuator: actuator PD와 tau_psi/tau_q 계산
                    |
                    +-> mdp/actions.py
                    |    raw action -> psi_target -> q_target
                    |
                    +-> mdp/observations.py
                    |    q/qdot -> psi/psi_dot observation
                    |
                    +-> mdp/rewards.py
                         ├─ tau_psi, psi_dot, psi_ddot penalty
                         ├─ actuator torque/velocity limit 초과 penalty
                         ├─ clipped raw actuator action-rate penalty
                         └─ q_target joint-position-limit penalty
                              -> flat_env_cfg.py의 RoK4RewardsCfg
                                   func + weight + params를 RewardTerm으로 구성

``rok4_adapt.py``, ``actions.py``, ``observations.py``, ``rewards.py`` 는 모두 ``rok4_lab`` 안의 로컬 파일이며,
Isaac Lab 원본 source를 수정하지 않는다. Isaac Lab의 부모 class와 mdp 함수는 import/상속/re-export해서 사용한다.

구조 요약
-------------------------------------------------

``RoK4FlatEnvCfg`` 는 환경 설정이고, ``RoK4RewardsCfg`` 는 reward term 묶음이다. Reward 구조는 두 가지
관계로 나누어 보면 덜 헷갈린다.

첫 번째는 config class의 상속/포함 관계이다.

.. code-block:: text

   RoK4FlatEnvCfg
     ├─ 상속: LocomotionVelocityRoughEnvCfg
     │          └─ 위치:
     │             IsaacLab/.../locomotion/velocity/velocity_env_cfg.py
     │
     └─ rewards: RoK4RewardsCfg()
                │
                └─ RoK4RewardsCfg
                     ├─ 상속: RewardsCfg
                     │          └─ 위치:
                     │             IsaacLab/.../locomotion/velocity/velocity_env_cfg.py
                     │
                     ├─ 부모 RewardsCfg에서 물려받은 reward terms
                     │    ├─ lin_vel_z_l2
                     │    ├─ ang_vel_xy_l2
                     │    ├─ flat_orientation_l2
                     │    └─ undesired_contacts
                     │
                     └─ RoK4RewardsCfg에서 새로 정의/override한 reward terms
                          ├─ termination_penalty
                          ├─ track_lin_vel_xy_exp
                          ├─ track_ang_vel_z_exp
                          ├─ feet_air_time
                          ├─ base_height_l2
                          ├─ feet_clearance
                          ├─ no_jumps
                          ├─ feet_slide
                          ├─ feet_touchdown_velocity
                          ├─ feet_contact_velocity (현재 None: 비활성)
                          ├─ feet_contact_force (현재 None: 비활성)
                          ├─ feet_touchdown_acc (현재 None: 비활성)
                          ├─ feet_flat_orientation_l2 (현재 None: 비활성)
                          ├─ feet_swing_roll_l2 (swing 중 roll 억제)
                          ├─ feet_swing_pitch_l2 (swing 중 pitch를 약하게 억제)
                          ├─ feet_stance_width_l2 (현재 None: 비활성)
                          ├─ feet_lateral_separation_l2 (signed lateral anti-cross)
                          ├─ stand_still_joint_deviation_l1
                          ├─ dof_pos_limits
                          ├─ joint_action_target_pos_limits
                          ├─ joint_deviation_hip
                          ├─ joint_deviation_hip_pitch
                          ├─ joint_deviation_torso
                          ├─ actuator_acc_l2
                          ├─ actuator_torques_l2
                          ├─ actuator_vel_l2
                          ├─ actuator_velocity_limits
                          ├─ actuator_torque_limits
                          ├─ action_rate_l2
                          └─ second_action_rate_l2

두 번째는 각 reward term이 실제 계산 함수를 참조하는 관계이다.

.. code-block:: text

   RoK4FlatEnvCfg
     └─ rewards = RoK4RewardsCfg()
          └─ RoK4RewardsCfg extends RewardsCfg
               └─ 각 reward term은 RewardTermCfg/RewTerm
                    └─ func=mdp.xxx
                         ├─ RoK4 로컬 mdp 함수
                         │    source/rok4_tasks/.../velocity/mdp/rewards.py
                         │
                         ├─ Isaac Lab 공통 mdp 함수
                         │    IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py
                         │
                         └─ locomotion velocity 전용 mdp 함수
                              IsaacLab/source/isaaclab_tasks/.../velocity/mdp/rewards.py

즉 ``RoK4FlatEnvCfg -> RoK4RewardsCfg`` 는 상속이 아니라 포함/사용 관계이고,
``RoK4RewardsCfg -> RewardsCfg`` 는 상속 관계이다. ``mdp/rewards.py`` 는 부모 클래스가 아니라 실제 reward
계산 함수가 들어 있는 함수 모음이다. RoK4는 로컬 ``rok4_tasks...velocity.mdp`` 를 import하며, 이 로컬 mdp는
Isaac Lab의 기존 locomotion mdp를 다시 export하고 RoK4 전용 reward 함수만 추가한다.

실제로는 다음처럼 이해하면 된다.

.. code-block:: text

   RoK4RewardsCfg는 RewardsCfg를 상속받아서 기본 reward term들을 물려받는다.
   RoK4에 필요한 reward는 mdp.xxx 함수를 직접 지정해 새로 정의하거나 override한다.
   RewardTermCfg/RewTerm은 mdp.xxx 함수, weight, params를 하나의 reward term으로 묶는다.

파일 배치 기준은 다음과 같다.

.. code-block:: text

   flat_env_cfg.py
     └─ 어떤 reward term을 쓸지, weight를 얼마로 둘지, 어떤 body/joint에 적용할지 결정

   velocity/mdp/rewards.py
     └─ reward raw value를 어떻게 계산할지 정의

따라서 ``joint_action_target_pos_limits``, ``actuator_torques_l2``, ``actuator_vel_l2``, ``actuator_acc_l2``,
``actuator_velocity_limits``, ``actuator_torque_limits``, ``action_rate_l2``,
``second_action_rate_l2`` 처럼
RoK4 전용 weighting을 쓰는 계산식은 ``velocity/mdp/rewards.py`` 에 두고, weight를 얼마로 사용할지는
``RoK4RewardsCfg`` 에 둔다.
이 term은 이전 action 두 개가 필요하므로 reset 직후 첫 두 policy step에서는 penalty를 0으로 둔다.

Action-rate reward의 action 기준
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

현재 RoK4Lab의 ``action_rate_l2`` 와 ``second_action_rate_l2`` 는 ``env.action_manager.action`` 을 기준으로
현재 action과 history의 차분을 만든다. Reward 함수는 별도 clamp를 하지 않는다. 하지만 표준 RoK4 RSL-RL
경로에서는 ``clip_actions=1.0`` wrapper가 ActionManager 이전에 정책 출력을 제한하므로 이 action buffer는
scale 적용 전 clipped raw policy action이다. Reward는 이 raw 차분의 weighted 제곱합을 계산한다.
Action scale은 적용하지 않지만 Hip Pitch/Knee에 해당하는
index ``2,3,8,9`` 의 제곱 오차에는 RoK4 전용 완화 weight ``0.5`` 를 적용한다.

``RoK4FlatPPORunnerCfg`` 에서 ``clip_actions = 1.0`` 을 명시하므로 Isaac Lab ``train.py`` 와 ``play.py`` 를
통해 실행할 때는 ``RslRlVecEnvWrapper`` 가 action을 먼저 ``[-1, 1]`` 로 clamp한다. 따라서 reward가 보는 action은
wrapper에서 ``[-1, 1]`` 로 잘린 raw actuator action이다.

ROBOTIS K1 Rev1의 reward 함수도 별도 clamp 없이 ``ActionManager.action`` 을 읽지만, K1 runner에는
``clip_actions`` 설정이 없어 기본값 ``None`` 을 사용한다. 따라서 K1은 unclipped raw action 차분, RoK4 표준
학습은 wrapper-clipped raw action 차분이라는 차이가 있다.

.. code-block:: text

   actor output
     -> clip_actions = 1.0
     -> clipped_raw_actuator_action
     ├─ last_action observation = clipped_raw_actuator_action
     ├─ action_rate_l2 / second_action_rate_l2
     └─ scaled_actuator_offset = clipped_raw_actuator_action * ROK4_ACTUATOR_ACTION_SCALE
          └─ psi_target = psi_default + scaled_actuator_offset
               └─ q_target = J * psi_target
                    └─ joint_action_target_pos_limits compares q_target with soft_joint_pos_limits

여기서 ``q_target`` 은 PhysX position drive에 직접 입력되는 값이 아니라 explicit actuator model의 입력이다.
``RoK4AdaptActuator.compute()`` 가 ``q_target`` 과 현재 joint state를 actuator 좌표로 변환해 ``tau_psi`` 를
계산·제한하고, ``tau_q = J^-T tau_psi`` 로 바꾼 joint effort만 PhysX에 전달한다. Actuator torque는 최대값의
90%에서 제어 및 reward 기준으로 제한하고, PhysX ``effort_limit_sim`` 은 factor를 적용하지 않은 joint mechanical
maximum을 최종 solver 안전 상한으로 사용한다.

기존 IsaacGym RoK4 코드는 ``actuator_actions_raw * action_scale`` 으로 만든 scaled ``actuator_actions`` 를
두 action-rate penalty에 넘겼다. 현재 RoK4Lab은 이 부분만 G1 방식으로 바꾸어 clipped raw action 차분을
사용한다. ``ROK4_ACTUATOR_ACTION_SCALE`` 값은 Gym의 좌우 다리 ``[0.4, 0.5, 1.25, 1.5, 0.75, 0.75]`` 와 torso
``0.4`` 를 그대로 유지하지만 actuator target 생성에만 사용한다. Observation의 ``last_action`` 과 두
action-rate reward가 같은 clipped raw action 좌표를 사용한다.

이 변경은 tensor 크기만 보면 이전 joint-space policy와 같지만 action, joint/actuator position, velocity의
의미가 다르다. 기존 joint-space checkpoint를 resume하면 잘못된 좌표 의미로 학습되므로 actuator-interface
checkpoint는 반드시 새 학습에서 생성한다.

Action target의 ``default_joint_pos`` 도 Gym의 활성 gait-ready pose와 동일하다. Hip pitch는
``-0.0924 rad``, knee pitch는 ``0.345 rad``, ankle pitch는 ``-0.253 rad`` 이며 나머지 관절은
``0.0 rad`` 이다. 이 joint pose를 ``psi_default = J^-1 q_default`` 로 변환한 값이 actuator action 중심이며,
``joint_deviation_*`` reward는 계속 원래 joint default pose를 기준으로 한다.

Reward와 Domain Randomization의 구분
----------------------------------------------------------------------------

``RoK4RewardsCfg`` 와 부모 ``RewardsCfg`` 는 reward term을 정의한다. 반면 foot friction, restitution,
base/upper/lower body mass, base/upper/lower COM, reset pose 같은 domain randomization 값은
``domain_randomization_cfg.py`` 에서 관리한다.

두 설정은 학습 결과에 모두 영향을 주지만 역할은 다르다.

.. list-table::
   :header-rows: 1

   * - 구분
     - 파일
     - 역할
   * - Reward
     - ``flat_env_cfg.py`` 의 ``RoK4RewardsCfg`` 및 ``self.rewards.xxx``
     - policy가 어떤 행동을 더 좋게 볼지 점수 함수를 정의
   * - DR
     - ``domain_randomization_cfg.py``
     - 물성, 질량, COM, 초기상태 perturbation을 바꿔 다양한 상황에서 버티게 함

예를 들어 발 미끄러짐은 ``feet_slide`` reward가 penalty로 줄이고, foot-ground 마찰 계수는
``domain_randomization_cfg.py`` 의 ``ROK4_STATIC_FRICTION_RANGE`` 와 ``ROK4_DYNAMIC_FRICTION_RATIO`` 가 정한다.
따라서 미끄러짐 문제를 볼 때는 reward와 DR을 함께 확인해야 하지만, 코드상 관리 위치는 분리되어 있다.

현재 material event는 모든 robot collision shape를 G1/Digit식 nominal ``mu_static=0.8``,
``mu_dynamic=0.6`` 으로 먼저 고정한다. 그 뒤 Foot material DR이 환경마다
``mu_static ~ Uniform(0.5, 0.9)`` 를 표본화하고
``mu_dynamic = 0.75 * mu_static`` 으로 계산한다. 같은 환경의 좌우 Foot은 동일한 material bucket을 사용하므로
마찰 DR 자체가 좌우 gait 비대칭을 만들지 않으며, dynamic friction은 항상 static friction 이하로 유지된다.
Play와 Teleop에서는 이 범위를 사용하지 않고 nominal ``0.8/0.6`` 으로 고정해 checkpoint를 재현성 있게 비교한다.

관절 물성은 학습 scene 생성 시 각 environment/joint에 대해 ``ROK4_STATIC_FRICTION``,
``ROK4_VISCOUS_FRICTION``, ``ROK4_ARMATURE`` 의 nominal 값을 각각 독립적인 ``Uniform(0.8, 1.2)`` 배율로
scale한다. 이는 PhysX joint property DR이며 actuator-space PD의 ``ROK4_ACTUATOR_KP/KD`` gain DR이 아니다.
Play와 Teleop에서는 이 joint-physics DR event를 제거해 nominal 관절 물성을 사용한다.

Reward와 PPO entropy/KL의 구분
----------------------------------------------------------------------

Reward term은 환경의 ``RewardManager`` 가 행동의 점수를 계산하는 항목이다. 반면 PPO의 entropy와 KL은
``rsl_rl`` optimizer가 policy distribution을 업데이트할 때 사용하는 학습 지표다. 따라서
``entropy_coef`` 나 ``desired_kl`` 은 ``RoK4RewardsCfg`` 의 reward weight가 아니다.

.. list-table:: Reward와 PPO 지표 비교
   :header-rows: 1

   * - 항목
     - 현재 값/위치
     - 역할
   * - reward weight
     - ``flat_env_cfg.py`` 의 ``RewTerm(weight=...)``
     - 환경 행동의 positive reward 또는 penalty 크기를 결정
   * - ``entropy_coef``
     - ``rsl_rl_ppo_cfg.py`` 의 ``0.002``
     - Gaussian policy의 exploration entropy를 유지하려는 optimizer 압력을 결정
   * - ``desired_kl``
     - ``rsl_rl_ppo_cfg.py`` 의 ``0.01``
     - old/new policy 차이를 기준으로 adaptive learning rate를 조절

RoK4 로컬 ``scripts/rsl_rl/rok4_ppo.py`` 는 adaptive schedule에서 이미 계산되는 mini-batch KL을 모아
TensorBoard의 ``Loss/kl``(iteration 평균)과 ``Loss/kl_max``(iteration 최대)로 기록한다. 현재 5 learning
epochs와 4 mini-batches를 사용하므로 iteration당 20개 KL 값을 집계한다. 이 로깅을 위해 Isaac Lab 또는
설치된 ``rsl_rl`` 파일을 수정하지 않는다.

Reward 계산 방식
--------------------------------------------------------

Isaac Lab의 ``RewardManager`` 는 각 reward term의 raw value를 계산한 뒤 ``weight`` 를 곱하고, 모든 term을
합산한다.

.. code-block:: text

   total_reward =
       sum(weight_i * reward_function_i(env, params_i))

따라서 positive weight는 행동을 유도하고, negative weight는 penalty로 작동한다.

RoK4 전용 Reward Terms
-------------------------------------------------------------

아래 term들은 ``RoK4RewardsCfg`` 에서 직접 정의하거나 부모 ``RewardsCfg`` 의 term을 RoK4용으로 override한
항목이다.

.. list-table::
   :header-rows: 1
   :widths: 18 15 10 27 30

   * - term
     - function
     - weight
     - params
     - 의미
   * - ``termination_penalty``
     - ``mdp.is_terminated``
     - ``-200.0``
     - 없음
     - episode가 termination되면 큰 penalty를 준다.
   * - ``track_lin_vel_xy_exp``
     - ``mdp.track_lin_vel_xy_yaw_frame_exp``
     - ``1.0``
     - ``command_name="base_velocity"``, ``std=0.5``
     - yaw-aligned robot frame에서 x/y 선속도 command를 추종하게 한다.
   * - ``track_ang_vel_z_exp``
     - ``mdp.track_ang_vel_z_world_exp``
     - ``1.0``
     - ``command_name="base_velocity"``, ``std=0.5``
     - world frame yaw angular velocity command를 추종하게 한다.
   * - ``base_height_l2``
     - ``mdp.base_height_relative_l2``
     - ``-1.0``
     - flat environment origin 기준 ``target_height=0.907 m``
     - root world Z에서 각 environment origin Z를 뺀 base height가 gait-ready IK 기준 높이와 달라지는 정도를 제곱 penalty로 만든다.
   * - ``feet_air_time``
     - ``mdp.FeetAirTimeTouchdownBiped``
     - ``2.0``
     - feet: ``L_Foot_Link``, ``R_Foot_Link``; ``target_air_time=0.50 s``, ``command_threshold=0.05 m/s``
     - 정확히 한 발이 first contact가 된 step에 완료된 air-time의 ``T - 0.50`` 을 한 번 지급하고 같은 유효 touchdown의 실제 ``T`` 를 metric으로 누적한다. 최대 보상 cap은 없으며 stateless 함수 버전도 비교용으로 남아 있다.
   * - ``feet_clearance``
     - ``mdp.feet_swing_clearance_exp``
     - ``0.2``
     - Foot body-origin target ``0.054 m`` (sole clearance ``0.050 m`` + origin offset ``0.004 m``), std ``0.04 m``, velocity scale ``0.50 m/s``
     - 선형 명령에서는 yaw frame의 command 방향으로 진행하는 swing Foot 속도만 ``tanh`` gate에 사용한다. 높이 Gaussian은 유지하며, 양발 진행 방향이 반대인 pure-yaw에서는 기존 XY 속도 크기를 사용한다. Air-time 길이는 touchdown reward가 별도로 담당한다.
   * - ``no_jumps``
     - ``mdp.desired_contacts``
     - ``-2.0``
     - feet: ``L_Foot_Link``, ``R_Foot_Link``; force ``threshold=1.0 N``
     - 최근 contact-force history에서 양발 모두 threshold를 넘는 접촉이 없을 때만 raw penalty ``1`` 을 반환한다. 정상 single stance와 toe-off는 허용하지만 양발 동시 flight phase를 억제한다.
   * - ``feet_slide``
     - ``mdp.feet_slide``
     - ``-0.2``
     - feet: ``L_Foot_Link``, ``R_Foot_Link``
     - 지면 접촉 중인 발이 미끄러지는 것을 줄인다.
   * - ``feet_touchdown_velocity``
     - ``mdp.FeetTouchdownVelocityL2``
     - ``-10.0``
     - safe landing velocity ``0.0 m/s``
     - 이전 policy step의 공중 발 world-Z 속도를 저장한다. First contact가 발생하면 하강속도 크기의 제곱 ``relu(-v_z_prev)^2`` 을 사건당 한 번 penalty로 만든다.
   * - ``feet_contact_velocity``
     - ``mdp.feet_contact_velocity_l2``
     - ``None`` (현재 비활성)
     - 실험값: landing height ``0.03 m``, approach threshold ``-0.6 m/s``, impact threshold ``0.0 m/s``, weight ``-10.0``
     - 기존 Gym의 접근/착지 velocity penalty 함수는 비교용으로 남아 있지만, 접근 구간까지 shaping해 보행 형태가 변한 실험 이후 활성 term에서 제외했다.
   * - ``feet_contact_force``
     - ``mdp.FeetContactForceL2``
     - ``None`` (현재 비활성)
     - 후속 비교 실험값: force limit ``1.2 x randomized body weight``, landing window ``0.10 s``, weight ``-0.05``
     - 함수는 비교용으로 남아 있으나 현재 학습 reward에는 연결하지 않는다.
   * - ``feet_contact_force_metrics``
     - ``mdp.FeetContactForceL2`` (``metric_only=True``)
     - ``1.0`` (함수 반환값은 항상 0)
     - landing window ``0.10 s``
     - Reward Manager가 매 step 호출하여 touchdown peak GRF를 집계하지만 reward 합에는 정확히 0을 더한다. TensorBoard ``Metrics/feet_touchdown/mean_peak_normal_force`` 만 기록한다.
   * - ``feet_touchdown_acc``
     - ``mdp.feet_touchdown_acc``
     - ``None`` (현재 비활성)
     - 실험값: acceleration threshold ``50 m/s^2``, weight ``-0.002``
     - ROBOTIS K1과 같은 함수는 남아 있지만 ``touchdownacc50`` 실험에서 MuJoCo와 실기 착지 충격이 개선되지 않아 비활성화했다. First-contact의 10 ms 판정 구간과 마지막 2 ms physics sample의 가속도 시점이 일치하지 않을 수 있다.
   * - ``feet_flat_orientation_l2``
     - ``mdp.feet_flat_orientation_l2``
     - ``None`` (현재 비활성)
     - 좌우 Foot body quaternion과 contact sensor
     - 함수는 접촉 중인 발의 local ``+Z`` 축을 world frame으로 회전한 뒤 X/Y 성분 제곱합을 계산한다. 현재는 world-up 수평 제약이 toe-off와 향후 경사면 적응을 방해하지 않도록 Reward Manager에 등록하지 않는다.
   * - ``feet_swing_roll_l2``
     - ``mdp.feet_swing_roll_l2``
     - ``-1.0``
     - 좌우 Foot body quaternion과 contact sensor
     - 공중에 있는 발의 local ``+Z`` 법선을 각 Foot 자체의 yaw frame으로 변환하고 lateral 성분의 제곱만 계산한다. 따라서 swing-foot의 안쪽/바깥쪽 roll을 억제하고 지지 발 자세는 제한하지 않는다.
   * - ``feet_swing_pitch_l2``
     - ``mdp.feet_swing_pitch_l2``
     - ``-0.1``
     - 좌우 Foot body quaternion과 contact sensor
     - 같은 yaw-removed sole normal의 forward 성분 제곱을 swing 발에만 적용한다. Roll 항의 1/10 가중치로 지속적인 toe-up을 약하게 억제하되 toe-off와 전후 swing 적응은 허용한다.
   * - ``feet_stance_width_l2``
     - ``mdp.feet_stance_width_l2``
     - ``None`` (현재 비활성)
     - 목표 ``0.21 m``, hard minimum ``0.18 m``; 직진 이동 command에서만 활성화
     - 함수는 좌우 Foot 위치 차이를 base yaw frame으로 회전해 lateral width를 계산한다. ``0.21 m`` 초과 폭은 coefficient ``0.5`` 로 억제하고, ``0.18 m`` 미만 또는 발 교차는 coefficient ``5.0`` 으로 강하게 억제한다. 현재 ``RoK4RewardsCfg`` 에서는 term을 ``None`` 으로 두어 Reward Manager에 등록하지 않는다.
   * - ``feet_lateral_separation_l2``
     - ``mdp.feet_lateral_separation_l2``
     - ``-2.0``
     - minimum width ``0.16 m``; feet 순서는 Left, Right
     - 좌우 Foot 위치 차이를 base yaw frame으로 회전한 뒤 signed lateral width ``y_left - y_right`` 를 유지한다. ``relu(0.16 - signed_width)^2`` 만 반환하므로 정상 ``0.21 m`` 폭과 넓은 폭은 제한하지 않고, 좁아질수록 penalty가 증가하며 좌우 Foot이 교차해 부호가 바뀌면 더 크게 작동한다. Command mask 없이 standing과 moving 모두에 적용된다.
   * - ``stand_still_joint_deviation_l1``
     - ``mdp.stand_still_joint_deviation_l1``
     - ``-0.2``
     - ``ROK4_JOINT_ORDER`` 전체 13관절, ``base_velocity.is_standing_env`` mask
     - RoK4 command generator가 standing으로 지정한 환경에서만 실제 joint position과 default joint position 차이의 절댓값 합을 penalty로 반환한다. 기존 Gym의 ``defaultPosStanding`` 에 대응한다.
   * - ``dof_pos_limits``
     - ``mdp.joint_pos_limits``
     - ``-1.0``
     - ``ROK4_JOINT_ORDER`` 에 포함된 전체 13관절, ``preserve_order=True``
     - 각 관절이 USD hard limit의 95%로 계산된 soft joint limit을 넘은 양을 합산해 억제한다.
   * - ``joint_action_target_pos_limits``
     - ``mdp.joint_action_target_pos_limits``
     - ``-0.001``
     - ``actuator_pos`` action의 mapped joint target과 ``ROK4_JOINT_ORDER`` 의 전체 13관절
     - Gym의 ``penalty_action_limits`` 에 대응한다. ADAPT mapping 뒤 목표 관절 위치가 95% soft limit을 넘은 양을 합산한다.

   * - ``joint_deviation_hip``
     - ``mdp.joint_deviation_l1``
     - ``-0.05``
     - ``.*_Hip_Yaw_Joint``, ``.*_Hip_Roll_Joint``
     - RoK4의 회전된 hip joint frame에서는 lateral foot placement가 이름상 yaw/roll 두 축의 조합으로 생성된다. ``-0.1`` 에서 절반으로 완화하여 ``flat_orientation_l2=-5.0`` 으로 상체 기울임은 억제하면서 다리가 옆으로 내딛을 자유를 준다. 완전히 끄지는 않아 과도한 hip 편차와 넓은 stance를 계속 억제한다.
   * - ``joint_deviation_hip_pitch``
     - ``mdp.joint_deviation_l1``
     - ``-0.005``
     - ``.*_Hip_Pitch_Joint``
     - 전후진 보폭을 만드는 핵심 관절을 yaw/roll과 같은 강도로 묶지 않으면서, default pose에서 과도하게 벗어나 다리 전체를 크게 휘두르는 전략을 약하게 억제한다. Swing phase나 무릎 굽힘을 직접 판정하는 reward는 아니다.
   * - ``joint_deviation_torso``
     - ``mdp.joint_deviation_l1``
     - ``-0.1``
     - ``Torso_Yaw_Joint``
     - torso yaw가 default pose에서 과도하게 벗어나지 않게 한다.
   * - ``actuator_acc_l2``
     - ``mdp.actuator_acc_l2``
     - ``-1.0e-8``
     - 전체 RoK4 actuator; index ``2,3,8,9`` weight ``0.5``
     - Isaac Lab의 joint acceleration을 ``J^-1`` 로 actuator acceleration으로 변환한다. 실제 ``rad/s^2`` 값이며 Gym의 undivided velocity difference와는 다르다.
   * - ``actuator_torques_l2``
     - ``mdp.actuator_torques_l2``
     - ``-2.0e-6``
     - 전체 RoK4 actuator; index ``2,3,8,9`` weight ``0.5``
     - ADAPT explicit actuator가 torque-limit clip 뒤 보관한 ``tau_psi`` 의 weighted squared sum이다.
   * - ``actuator_vel_l2``
     - ``mdp.actuator_vel_l2``
     - ``-1.0e-4``
     - 전체 RoK4 actuator; index ``2,3,8,9`` weight ``0.5``
     - ``psi_dot = J^-1 q_dot`` 의 weighted squared sum으로 Gym ``penalty_joint_vel`` 에 대응한다.
   * - ``actuator_velocity_limits``
     - ``mdp.actuator_velocity_limits``
     - ``-0.001``
     - ``velocity_limit_factor=0.9`` 를 적용한 actuator velocity limits
     - ``abs(psi_dot)`` 가 설정된 velocity limit을 넘은 양을 합산한다.
   * - ``actuator_torque_limits``
     - ``mdp.actuator_torque_limits``
     - ``-1.0e-5``
     - ``torque_limit_factor=0.9`` 를 적용한 actuator torque limits
     - clip 전 요청 ``tau_psi`` 가 설정된 torque limit을 넘은 양을 합산해 saturation 요구를 표시한다.
   * - ``action_rate_l2``
     - ``mdp.action_rate_l2``
     - ``-0.01``
     - clipped raw action 전체 13차원; index ``2,3,8,9`` weight ``0.5``
     - ``a_t - a_{t-1}`` 의 weighted 제곱합을 줄인다. Action scale은 적용하지 않는다.
   * - ``second_action_rate_l2``
     - ``mdp.second_action_rate_l2``
     - ``-0.005``
     - clipped raw action 전체 13차원; index ``2,3,8,9`` weight ``0.5``
     - raw action의 2차 차분, 즉 ``a_t - 2 a_{t-1} + a_{t-2}`` 의 weighted 제곱합으로 action curvature를 완화한다.
       reset 직후 첫 두 policy step은 history가 부족하므로 0으로 처리한다.

Gym과 Lab의 action-limit 신호 차이
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gym과 현재 Lab 구현 모두 actuator action을 ADAPT 변환한 mapped joint target을 검사한다. 현재 Lab의
``RoK4ActuatorPositionAction.processed_actions`` 가 ``q_target = J psi_target`` 을 보관하므로
``joint_action_target_pos_limits`` 는 실제 PD command에 대응하는 joint target을 검사한다.

Gym과 Lab의 torque/velocity 신호 차이
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

기존 Gym 코드와 현재 Lab 코드는 모두 ADAPT transmission 변환 전 actuator torque/velocity를 penalty에 사용한다.
현재 Lab custom actuator는 ``applied_actuator_effort`` 를 직접 노출하고 joint velocity를 ``J^-1`` 로 변환한다.
따라서 torque/velocity reward 좌표계와 ``0.5`` weighting은 Gym과 동일하다. Acceleration만 현재 Lab에서
``J^-1 q_ddot`` 의 실제 acceleration을 사용하므로 Gym의 ``psi_dot_t - psi_dot_(t-1)`` 과 정의가 다르다.

부모 RewardsCfg에서 상속받고 RoK4에서 수정한 Terms
-----------------------------------------------------------------------------------------

아래 term들은 ``RewardsCfg`` 에서 기본으로 제공되며, ``RoK4FlatEnvCfg.__post_init__()`` 에서 weight나 body/joint
대상이 RoK4에 맞게 수정된다.

.. list-table::
   :header-rows: 1
   :widths: 18 15 12 24 31

   * - term
     - function
     - current weight
     - RoK4 설정
     - 의미
   * - ``lin_vel_z_l2``
     - ``mdp.lin_vel_z_l2``
     - ``-0.2``
     - 기본 robot root
     - base가 z 방향으로 튀는 움직임을 줄인다.
   * - ``ang_vel_xy_l2``
     - ``mdp.ang_vel_xy_l2``
     - ``-0.05``
     - 기본 robot root
     - roll/pitch angular velocity를 줄인다.
   * - ``flat_orientation_l2``
     - ``mdp.flat_orientation_l2``
     - ``-5.0``
     - 기본 robot root
     - projected gravity의 X/Y 성분 제곱합으로 몸이 기울어지는 것을 줄인다. ``-1.0`` 과 ``-2.0`` 실험 뒤 ``-5.0`` 으로 크게 강화하여 lateral hopping이 과도한 body roll 전략인지 확인하고, 자연스러운 체중 이동과 속도 추종을 과도하게 제한하는지도 함께 관찰한다.
   * - ``undesired_contacts``
     - ``mdp.undesired_contacts``
     - ``-1.0``
     - ``Base_Link``, ``Upper_Body_Link``; ``threshold=1.0``
     - 발이 아닌 base/upper body 접촉을 penalty로 처리한다.

Reward Function 요약
------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - function
     - 위치
     - 계산 의미
   * - ``is_terminated``
     - Isaac Lab 공통
     - termination manager의 terminated flag를 reward term으로 반환한다.
   * - ``track_lin_vel_xy_yaw_frame_exp``
     - locomotion mdp
     - yaw frame x/y 선속도 command error에 ``exp(-error / std^2)`` 를 적용한다.
   * - ``track_ang_vel_z_world_exp``
     - locomotion mdp
     - world z축 angular velocity command error에 ``exp(-error / std^2)`` 를 적용한다.
   * - ``base_height_relative_l2``
     - RoK4 local mdp
     - flat environment origin 기준 root 높이와 ``0.907 m`` 목표의 제곱 오차를 반환한다.
   * - ``feet_air_time_touchdown_biped``
     - RoK4 local mdp
     - 정확히 한 발이 first contact가 된 순간 ``last_air_time - target_air_time`` 을 한 번 반환하는 상한 없는 stateless 함수다.
   * - ``feet_swing_clearance_exp``
     - RoK4 local mdp
     - 선형 명령에서는 유효 swing Foot별 ``tanh(relu(v_foot,yaw·unit(command_xy))/0.50) exp(-(h-0.054)^2/0.04^2)`` 를 평균한다. Pure-yaw에서는 ``v_xy`` 크기를 사용하고 standing에서는 0이다.
   * - ``desired_contacts``
     - Isaac Lab 공통
     - 지정 body 중 하나라도 최근 force history에서 threshold를 넘으면 0, 모두 접촉하지 않으면 1을 반환한다. Negative weight와 함께 양발 flight penalty로 사용한다.
   * - ``feet_slide``
     - locomotion mdp
     - 접촉 중인 foot의 horizontal linear velocity norm을 penalty로 계산한다.
   * - ``feet_contact_velocity_l2``
     - RoK4 local mdp
     - 비활성 비교용 함수다. 접촉 전에는 ``I(h<0.03) I(v_z<-0.6) v_z^2``, first contact에서는 ``I(v_z<0) v_z^2`` 를 발별로 합산한다.
   * - ``feet_contact_force_l2``
     - RoK4 local mdp
     - 비활성 비교용 함수다. 5-sample 법선력 history의 peak에 대해 ``sum(w_contact (relu(F_peak-1000)/1000)^2)`` 를 반환한다. ``w_contact`` 는 stance ``0.15``, first contact ``1.0`` 이다.
   * - ``FeetTouchdownVelocityL2``
     - RoK4 local mdp
     - 이전 policy sample의 공중 발 하강속도 ``s=relu(-v_z,previous)`` 를 저장하고, first contact에서만 ``sum(s^2)`` 를 반환한다. Reward Manager reset 시 history를 무효화해 초기 접촉을 제외한다.
   * - ``feet_touchdown_acc``
     - RoK4 local mdp
     - first-contact인 발에 대해서만 ``relu(||a_foot,w|| - threshold)`` 를 합산하는 ROBOTIS K1 호환 함수다. 현재 Reward Manager term은 ``None`` 이며 함수와 단위 테스트만 비교용으로 유지한다.
   * - ``lin_vel_z_l2``
     - Isaac Lab 공통
     - base body-frame z velocity squared.
   * - ``ang_vel_xy_l2``
     - Isaac Lab 공통
     - base body-frame roll/pitch angular velocity squared sum.
   * - ``flat_orientation_l2``
     - Isaac Lab 공통
     - projected gravity의 x/y component squared sum으로 몸 기울기를 penalty화한다.
   * - ``actuator_torques_l2``
     - RoK4 로컬 mdp
     - actuator-space applied torque weighted squared sum. Index ``2,3,8,9`` 는 weight ``0.5``.
   * - ``actuator_vel_l2``
     - RoK4 로컬 mdp
     - ``J^-1 q_dot`` actuator velocity의 weighted squared sum. Index ``2,3,8,9`` 는 weight ``0.5``.
   * - ``actuator_acc_l2``
     - RoK4 로컬 mdp
     - ``J^-1 q_ddot`` actuator acceleration의 weighted squared sum. Index ``2,3,8,9`` 는 weight ``0.5``.
   * - ``actuator_velocity_limits``
     - RoK4 로컬 mdp
     - ``velocity_limit_factor`` 를 적용한 actuator velocity limit 초과량.
   * - ``actuator_torque_limits``
     - RoK4 로컬 mdp
     - clip 전 actuator torque command가 ``torque_limit_factor`` 적용 limit을 넘은 양.
   * - ``action_rate_l2``
     - RoK4 로컬 mdp
     - 현재 raw action과 이전 action 차이의 weighted squared sum. 함수 내부 clamp는 없지만 표준 RoK4 runner가 입력을 먼저 ``[-1,1]`` 로 제한한다. Index ``2,3,8,9`` 는 weight ``0.5``.
   * - ``second_action_rate_l2``
     - RoK4 로컬 mdp
     - raw action의 2차 차분 weighted squared sum. 표준 RoK4 runner 경로에서는 wrapper-clipped raw action이다. Index ``2,3,8,9`` 는 weight ``0.5``.
   * - ``joint_deviation_l1``
     - Isaac Lab 공통
     - 현재 joint position과 default joint position 차이의 absolute sum.
   * - ``stand_still_joint_deviation_l1``
     - RoK4 로컬 mdp
     - ``base_velocity.is_standing_env`` 가 true일 때 전체 13관절의 default-pose absolute error sum.
   * - ``joint_pos_limits``
     - Isaac Lab 공통
     - soft joint position limit을 넘은 정도를 합산한다.
   * - ``joint_action_target_pos_limits``
     - RoK4 로컬 mdp
     - ``actuator_pos`` action이 ADAPT mapping한 joint target의 soft joint position limit 초과량.
   * - ``undesired_contacts``
     - Isaac Lab 공통
     - 지정 body의 contact force가 threshold를 넘은 contact 개수를 penalty로 반환한다.

Command와 Reward의 연결
--------------------------------------------------------------

현재 RoK4 flat command range는 다음과 같다.

.. code-block:: python

   lin_vel_x = (-0.3, 0.85)
   lin_vel_y = (-0.3, 0.3)
   ang_vel_z = (-0.6, 0.6)

``RoK4PeriodicFreezeVelocityCommand`` 는 episode reset마다 환경 역할을 다음 확률로 표본화한다.

1. 35% ``mixed``: ``vx``, ``vy``, ``wz`` 를 모두 표본화한다.
2. 5% ``standing``: episode 동안 exact-zero command를 유지한다.
3. 5% ``walking``: planar command norm ``0.10 m/s`` 이상만 사용하고 freeze하지 않는다.
4. 20% ``x``: ``vx=+/-Uniform(0.10, 0.30) m/s`` 저속 대칭 전후진만 표본화한다.
5. 5% ``fast_forward``: ``vx=Uniform(0.30, 0.85) m/s`` 고속 전진만 표본화한다.
6. 10% ``y``: lateral 좌/우 command만 표본화한다.
7. 10% ``yaw``: 시계/반시계 yaw-rate command만 표본화한다.
8. 10% ``x_yaw``: ``vy=0`` 인 저속 대칭 전후진+회전 command를 표본화한다.

코드에서는 command 축과 직접 대응하는 ``x/y/yaw/x_yaw`` 이름을 사용한다. 각 단일 축 역할은 양/음 부호를
50:50 확률로 표본화하며 ``x_yaw`` 의 두 부호는 독립적으로 표본화한다. ``x`` 와 ``x_yaw`` 의 ``vx`` 는
절댓값 ``0.10~0.30 m/s`` 로 대칭이고, ``fast_forward`` 가 ``0.30~0.85 m/s`` 를 별도로 담당한다. ``vy`` 의
최소 절댓값은 ``0.10 m/s``, ``wz`` 는 ``0.10 rad/s`` 이다. 4096개 환경에서 20% ``x`` 역할은 평균 약 819개이고,
각 부호는 평균 약 410개다. 나머지 10% 전용 역할은 각각 평균 약 410개다.

``mixed/x/fast_forward/y/yaw/x_yaw`` 의 비율 합 90%는 ``10.0 s`` cycle과 환경별 ``Uniform(1.5, 3.0) s`` freeze duration을
사용한다. Random initial phase와 독립 duration 때문에 같은 PPO batch에서 moving, standing, transition을
동시에 관측한다. Episode 중 command를 다시 표본화할 때는 역할을 유지하며, reset 때 역할 자체를 다시 무작위로
뽑는다. 부모의 별도 standing 확률은 ``rel_standing_envs=0.0`` 으로 비활성화한다. 작은 non-zero mixed command는
standing으로 바꾸지 않고 연속적인 저속 이동 목표로 유지한다.

여기서 random initial phase는 global simulation 시작 시 한 번만 주는 값이 아니다. 각 environment가 reset될
때마다 freeze phase를 ``Uniform(0, 10) s`` 로 다시 표본화하므로 reset 직후 freeze로 시작할 수도 있다. 학습
episode timeout은 환경별 ``20 s`` 이며, early termination이 난 환경만 episode counter, episode role, freeze
phase를 다시 시작한다. 예를 들어 global time ``3 s`` 에 reset된 환경은 이후 20초를 생존하면 약 ``23 s`` 에
timeout된다. 초기에는 episode counter가 같지만 early reset이 누적되면서 환경별 episode phase가 자연스럽게
달라진다.

부모 task에서 상속한 training ``push_robot`` 은 freeze와 다른 환경별 ``interval`` timer다. Reset마다 다음
push 시간을 ``Uniform(10, 15) s`` 로 표본화하고, world-frame root x/y velocity에 각각 ``-0.5~0.5 m/s`` 를
추가한다. 따라서 push는 이동 command 중에도, exact-zero freeze 중에도, 전환 부근에도 발생할 수 있다. Freeze
중 push가 들어와도 command와 ``is_standing_env`` 는 zero/true로 유지되므로 velocity-tracking reward와
``stand_still_joint_deviation_l1`` 은 외란에서 정지 자세로 복원하는 행동을 평가한다. Contact reward와
``no_jumps`` 는 같은 구간의 실제 발 접촉 상태를 그대로 평가한다. Play/Teleop에서는 이 자동 interval event를
끄고 수동 Push Test UI를 사용한다.

Periodic freeze와 ``standing`` 역할은 command를 ``[0, 0, 0]`` 으로 만드는 동시에 ``is_standing_env=True`` 를
설정한다. ``stand_still_joint_deviation_l1`` 은 command 크기를 다시 판정하지 않고 이 mask를 직접 사용하여
전체 13관절을 default pose 근처로 유지한다. 현재 weight는 ``-0.2`` 이다. ``-1.0`` 실험은 안정적인 양발
standing을 만들었지만 feet-air-time 감소와 foot-slide 증가가 관측되어 보행 자유도를 회복하도록 완화했다.

이 command는 ``track_lin_vel_xy_exp``, ``track_ang_vel_z_exp``, ``feet_air_time`` 에 직접 영향을 준다. 특히
``FeetAirTimeTouchdownBiped`` 는 x/y command norm이 ``0.05`` 이하이면 reward를 0으로 만든다. 즉 거의 정지
명령에서는 stepping reward가 강하게 작동하지 않는다.

``UniformVelocityCommand`` 가 반환하는 ``[lin_vel_x, lin_vel_y, ang_vel_z]`` 는 robot base frame 기준
command다. 따라서 양의 ``lin_vel_x`` 는 로봇이 현재 바라보는 방향의 전진, 양의 ``lin_vel_y`` 는 로봇 기준
좌측 이동, 양의 ``ang_vel_z`` 는 z축 주위 반시계 방향 회전을 뜻한다. 선속도 tracking 함수는 실제 world 선속도에서
roll/pitch를 제외한 yaw 회전만 역변환한 gravity-aligned yaw frame 속도와 x/y command를 비교한다. 반면 현재
각속도 tracking 함수는 command의 z 성분과 world-frame ``root_ang_vel_w[:, 2]`` 를 비교한다. 평평하고 직립한
상태에서는 두 z축이 거의 같지만, 몸체가 크게 기울면 완전히 동일한 표현은 아니다.

RoK4는 부모 G1-style heading mode를 사용하지 않는다. ``heading_command=False``, ``rel_heading_envs=0.0``,
``heading=None`` 으로 설정하고 ``ang_vel_z=(-0.6, 0.6) rad/s`` 에서 목표 ``wz`` 를 직접 표본화한다. 따라서
학습, Gym, Teleop, ROS ``cmd_vel`` 과 실물 적용에서 policy가 받는 세 번째 command는 모두 같은 의미의 목표
yaw angular velocity다. Navigation에서 목표 world heading이 필요하면 policy 외부의 상위 controller가
heading error를 ``wz`` 로 변환해 이 동일한 velocity interface에 전달한다.

현재 ``feet_air_time`` 은 느리고 긴 step을 유도하기 위해 ``target_air_time=0.50 s``, ``weight=2.0`` 을
사용한다. 정확히 한 발이 first contact가 된 policy step에 그 발의 완료된 ``last_air_time`` 을 읽고
``last_air_time - 0.50`` 을 한 번 반환한다. 양발이 같은 step에 first contact가 되거나 planar command norm이
``0.05 m/s`` 이하이면 0이다. 따라서 ``T=0.30 s`` 는 raw ``-0.20``, ``T=0.50 s`` 는 ``0``, ``T=0.75 s`` 는
raw ``+0.25`` 이다. Reward Manager가 적용하는 pre-``dt`` weighted event는 각각 raw 값에 ``2.0`` 을 곱한다.
Policy interval ``0.01 s`` 까지 포함한 touchdown-error 기울기는 ``0.02`` 로, K1의 ``1.0 * 0.02`` 와 같다.

이 signed touchdown shaping은 짧은 잔걸음을 단순히 적게 보상하는 대신 직접 penalty화한다. 반면
최대 보상 air-time cap이 없으므로 ``0.75 s`` 를 초과한 완료 swing도 더 큰 양수를 받는다. 한 발을 오래 드는
전략은 velocity tracking, ``no_jumps`` 와 기타 gait term이 함께 제한해야 한다.
이 sparse event term의 TensorBoard 값은 이전 dense 또는 squared touchdown term과 직접 비교할 수 없다.

현재 활성 term은 stateful ``FeetAirTimeTouchdownBiped`` class다. Reward Manager가 기록하는
``Episode_Reward/feet_air_time`` 은 episode 동안 합산된 weighted reward이므로 실제 평균 air time 자체와
같지 않다. Class는 touchdown event의 ``last_air_time`` 합과 event count를 GPU tensor로 누적하고 reset 때
``Metrics/feet_touchdown/mean_air_time`` [s]를 기록한다.

``base_height_l2`` 는 평지에서 height scanner 없이 각 environment origin을 지면 기준으로 사용한다.

.. math::

   p_{base-height}
   = \left(z_{base,w} - z_{origin,w} - 0.907\right)^2

Raw 값에는 ``weight=-1.0`` 과 policy ``dt`` 가 적용된다. 목표는 standing에서만이 아니라 전체 보행에 적용되므로
몸통의 자연스러운 상하 진동도 작은 penalty를 받는다. 첫 실험에서는 높이를 완전히 고정하는 강한 제약이 아니라
평균 자세가 크게 주저앉거나 과도하게 올라가는 것을 막는 약한 기준으로 사용한다.

``feet_clearance`` 는 touchdown event가 아니라 swing 중 매 policy step 계산하는 dense positive reward다. 유효한
Foot 집합을 ``S`` 라고 하면 raw reward는 다음과 같다.

.. math::

   r_{clear}
   = \frac{1}{|S|}
     \sum_{i \in S}
     \tanh\left(\frac{\|v_{i,xy}\|}{0.50}\right)
     \exp\left(-\frac{(h_i-0.054)^2}{0.04^2}\right)

여기서 ``h_i = z_FootLink,i,w - z_origin,w`` 이고 ``v_i,xy`` 는 Foot body의 world-frame planar speed다.
``S`` 에는 ``current_air_time > 0`` 이고 command generator의 ``is_standing_env=False`` 인 발만 포함된다.
유효한 발이 없으면 0이다. 두 발이 동시에 유효해도 합이 아니라 평균하므로 raw 최대값은 1 미만이다. Weight는
``+0.2`` 이다.

현재 ``Foot_Link`` 원점은 약 8 mm 발 두께의 가운데에 있어 실제 sole보다 약 4 mm 높다. 따라서 이 ``h_i`` 는
정확한 collision-point clearance가 아니라 평지용 body-origin proxy다. 목표 ``0.054 m`` 는 원하는 sole
clearance ``0.050 m`` 에 이 origin offset ``0.004 m`` 를 더한 값이다. ``std=0.04 m`` 기준에서 4 mm offset의
차이는 target 자체에 명시적으로 반영했으며, 평지에서는 environment origin이 지면 높이와 같으므로 별도 height
scanner를 추가하지 않는다. 속도 gate는 높이 목표에 도달했더라도 거의 정지한 발이 점수를 받는 것을 막는다. 예를 들어
``v_xy=0.10 m/s`` 에서는 ``tanh(0.2) ~= 0.197``, ``0.20 m/s`` 에서는 ``tanh(0.4) ~= 0.380``,
``0.50 m/s`` 에서는 ``tanh(1) ~= 0.762`` 이며 고속에서는 부드럽게 1에 포화된다. 따라서 이전 ``0.20 m/s``
scale보다 느린 swing Foot의 clearance shaping을 약하게 만든다.

``no_jumps`` 는 ``mdp.desired_contacts`` 를 ``weight=-2.0`` 과 force ``threshold=1.0 N`` 으로 사용한다.
최근 contact-force history에서 좌우 Foot 모두 threshold를 넘지 못한 경우에만 raw value ``1`` 을 반환하므로,
Reward Manager에서는 해당 양발 flight interval에 음의 점수가 적용된다. 한 발이라도 접촉 중이면 0이므로 정상
single stance와 toe-off는 직접 억제하지 않는다. 또한 이 term만으로 좌우 교대나 최대 single-stance 시간을
강제하지는 않는다. 두 velocity-tracking term의 최대 pre-``dt`` 합 ``+2.0`` 을 flight 구간에서 상쇄하도록
``-2.0`` 을 첫 실험값으로 사용한다.

``feet_swing_roll_l2`` 는 ``weight=-1.0`` 로 활성화되어 swing 중 발바닥이 안쪽 또는 바깥쪽으로 말리는
현상을 억제한다. 각 Foot의 local ``+Z`` 법선을 world frame으로 회전한 뒤 그 Foot 자체의 yaw frame으로 옮기고,
lateral 성분 ``n_y^2`` 만 penalty로 반환한다. 새 ``feet_swing_pitch_l2`` 는 같은 벡터의 forward 성분
``n_x^2`` 를 사용하되 ``weight=-0.1`` 만 적용하여 지속적인 toe-up을 약하게 억제한다. Contact sensor의
``current_contact_time`` 이 0인 발에만 적용하고 swing 발 수로 평균하므로 지지 발은 두 항 모두 0이다.
Pitch 가중치는 roll의 1/10이므로 비활성 ``feet_flat_orientation_l2`` 처럼 발 전체를 항상 world-up에
고정하지 않으며, 필요한 toe-off와 sagittal swing을 더 높은 우선순위의 tracking/clearance 항이 사용할 수 있다.

Contact sensor는 ``update_period=0.002 s`` 와 ``history_length=self.decimation=5`` 를 사용한다. 따라서
``no_jumps``, ``feet_slide``, ``undesired_contacts``, ``illegal_body_contact`` 처럼 ``net_forces_w_history`` 를
검사하는 항목은 최근 policy interval의 5개 physics contact sample을 확인한다. 비활성 비교용
``feet_contact_force_l2`` 도 ground-filtered ``force_matrix_w_history`` 를 같은 방식으로 검사한다. 반면
활성 ``FeetAirTimeTouchdownBiped`` class와 비교용 ``feet_air_time_touchdown_biped`` 함수는 모두
``compute_first_contact(env.step_dt)`` 와 완료된 ``last_air_time`` 을 사용한다.
이 contact-force history는 policy observation history와 별개의 buffer이다.

현재 soft-landing 실험은 ``FeetTouchdownVelocityL2`` 만 활성화한다.
Velocity class는 각 발의 이전 policy step world-Z 속도와 접촉 여부를 저장하고, 이전 sample이
공중이었던 발에 first contact가 발생할 때만 다음 raw penalty를 반환한다.

.. math::

   P_{landing}=\sum_i I_{first,i} I_{air,previous,i}
   \left[\max(0,-v_{z,previous,i})\right]^2

Reward Manager는 여기에 weight ``-10.0`` 과 ``dt=0.01 s`` 를 적용한다. 따라서 접근 중과 계속된 stance에는
직접 영향을 주지 않는다. 별도 dead zone은 없지만 L2 특성상 작은 하강속도에는 작은 penalty만 생긴다. Reset
직후에는 이전 sample이 없으므로 초기 접촉을 제외한다.

현재 검증 기준은
``2026-08-12_23-45-39_privileged250_gain240_160_80_air050_w2_tdvel10_ar01_ar2_005_noforce_delay4ms_fresh20k``
의 ``model_19999.pt`` 다. 마지막 checkpoint에서 평균 착지 직전 하강속도는 약 ``0.044 m/s`` 였고 velocity
tracking과 평균 완료 air time은 유지되었다. 학습 checkpoint는 Isaac Lab log directory에 보존하며 Git에는
포함하지 않는다.

``FeetContactForceL2`` class의 shaping term은 ``feet_contact_force=None`` 이다. 별도
``feet_contact_force_metrics`` term은 ``metric_only=True`` 와 weight ``1.0`` 으로 매 step 실행되지만 함수가
항상 0을 반환하므로 total reward와 policy gradient에 영향을 주지 않는다. 이 logging-only term은 touchdown
후 ``0.10 s`` 동안의 peak GRF를 집계해 TensorBoard에 기록한다.

Air-time과 touchdown velocity class는 episode 누적값과 event count를 GPU tensor로 유지하고,
환경 reset 때 ``Metrics/feet_touchdown/*`` 의 event-weighted 평균을 기록한다. 기존 Gym 호환 stateless
``feet_contact_velocity_l2`` 와 ``feet_contact_force_l2`` 함수는 비교용으로 남아 있지만 현재 Reward Manager
term에는 연결하지 않는다.

Play 환경은 현재 standing 검증을 위해 ``lin_vel_x=0.0 m/s`` 를 사용하고 lateral/yaw command도 0으로 고정한다. 별도의 Teleop 환경은
긴 resampling interval을 사용하고, gamepad 또는 keyboard의
``[lin_vel_x, lin_vel_y, ang_vel_z]`` 를 동일한 base-frame command buffer에 직접 기록한다. 입력은 학습 범위
``(-0.3, 0.85) m/s``, ``(-0.3, 0.3) m/s``, ``(-0.6, 0.6) rad/s`` 안으로 scale된다. 이는 inference command
source만 바꾸며 reward 정의나 학습 checkpoint 자체를 변경하지 않는다. Play와 Teleop에서는 학습 전용 periodic
freeze를 비활성화한다.

Termination과 Reward의 연결
-------------------------------------------------------------------

RoK4 flat task는 Isaac Lab의 ``TerminationsCfg`` 를 수정하지 않고 로컬 ``RoK4TerminationsCfg`` 로 상속한다.
부모의 ``time_out`` 은 유지하고 ``base_contact`` 는 비활성화한 뒤 ``illegal_body_contact`` 를 추가한다.

.. code-block:: python

   class RoK4TerminationsCfg(TerminationsCfg):
       base_contact = None
       illegal_body_contact = DoneTerm(
           func=mdp.illegal_contact,
           params={
               "sensor_cfg": SceneEntityCfg(
                   "contact_forces",
                   body_names=ROK4_ILLEGAL_CONTACT_BODY_NAMES,
               ),
               "threshold": 1.0,
           },
       )

``ROK4_ILLEGAL_CONTACT_BODY_NAMES`` 에는 ``Base_Link``, ``Upper_Body_Link`` 와 좌우 ``Hip_Yaw``, ``Hip_Roll``,
``Thigh``, ``Calf``, ``Ankle_Pitch``, ``Ankle_Roll`` link가 포함된다. 정상 접촉 body인 ``L_Foot_Link`` 와
``R_Foot_Link`` 만 제외된다. 선택된 body 중 하나라도 contact force가 ``1.0 N`` 을 넘으면 episode가 종료된다.

``mdp.is_terminated`` 는 non-timeout termination을 감지하므로 ``illegal_body_contact`` 가 발생하면
``termination_penalty=-200.0`` 가 함께 작동한다. 반면 부모에서 상속한 정상 ``time_out`` 에는 이 penalty가
적용되지 않는다. 여러 body가 동시에 접촉해도 termination flag는 boolean이므로 termination penalty는 한 번만
적용된다. Contact sensor는 상대 물체 종류를 구분하지 않은 net contact force를 사용하므로 self-collision으로
선택 body에 큰 접촉력이 생겨도 종료될 수 있다.

Touchdown air-time 로그 해석
------------------------------------------------------

Directional-gait 기준 run
``2026-08-03_14-58-46_touchdown_air_symmetric_x_fastforward_fresh20k`` 의
``Episode_Reward/feet_air_time`` 은 다음처럼 변했다.

.. list-table:: Touchdown feet-air-time checkpoint 값
   :header-rows: 1

   * - checkpoint
     - logged reward
   * - ``model_9999.pt``
     - ``0.00278``
   * - ``model_14999.pt``
     - ``0.00244``
   * - ``model_19999.pt``
     - ``0.00297``
   * - 마지막 500 iteration 평균
     - ``0.00272``

이 scalar는 발의 평균 air-time 초 단위 값이 아니다. Reward Manager가 기록하는 episode-normalized 값에는
policy ``dt=0.01 s``, reward ``weight=2.0``, 유효 touchdown 빈도, 완료된 air-time, planar moving-command
mask가 함께 들어간다. 개념적으로 다음 곱에 가깝다.

.. math::

   \bar r_{air} \approx 0.01 \cdot 2.0 \cdot
   f_{touchdown} \cdot
   \mathbb{E}\left[T_{air}-0.50\right] \cdot p_{moving}

규칙적인 교대 보행에서는 touchdown 빈도와 step air-time이 서로 반비례할 수 있으므로 이 scalar 하나로
``T_air`` 을 역산할 수 없다. 또한 10k, 15k, 20k 값이 단조 증가하지 않아도 보행이 악화되었다는 뜻이 아니다.
현재 구현은 정확한 평균 비교를 위해 touchdown ``last_air_time`` 과 착지 직전 수직속도를 event count로 나눈
물리 단위 TensorBoard metric을 함께 기록한다. 중앙값/상위 백분위와 초당 touchdown 횟수는 아직 집계하지 않는다.

* ``Metrics/feet_touchdown/mean_pre_touchdown_vertical_speed`` [m/s]
* ``Metrics/feet_touchdown/mean_air_time`` [s]
* ``Metrics/feet_touchdown/mean_peak_normal_force`` [N]

``mean_peak_normal_force`` 는 기존 TensorBoard tag 호환을 위해 이름을 유지한다. 실제 집계값은 filtered
world-frame 접촉 합력 ``||[F_x,F_y,F_z]||`` 의 landing-window peak다. ``feet_contact_force_metrics`` 는
이 값만 기록하며 reward에는 0을 반환한다.

현재 설계 의도
------------------------------------------------------

현재 reward는 다음 목표를 동시에 가진다.

.. code-block:: text

   1. command velocity를 따라간다.
   2. upright 자세를 유지한다.
   3. 한 발씩 드는 biped stepping pattern을 만든다.
   4. 양발이 동시에 공중에 뜨는 flight phase를 줄이되 정상 single stance는 허용한다.
   5. standing과 moving 모두에서 좌우 Foot의 signed lateral separation 하한만 두어 다리 교차를 억제한다.
   6. foot-flat orientation과 목표 stance-width 함수는 구현되어 있지만 현재 비활성이다.
   7. 발 미끄러짐을 줄인다.
   8. torque, joint acceleration, action rate, second action rate를 줄여 움직임을 부드럽게 한다.
   9. ankle limit, hip/torso deviation을 제한한다.
   10. zero command에서 전체 13관절을 default standing pose 근처로 유지한다.
   11. Foot를 제외한 body 접촉을 실패 종료로 처리한다.

튜닝 시 우선 확인할 항목
---------------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - 현상
     - 먼저 볼 항목
   * - 너무 자주 넘어진다
     - ``termination_penalty``, ``illegal_body_contact``, self-collision, ``flat_orientation_l2``, 초기 자세
   * - 거의 걷지 않고 버틴다
     - ``track_lin_vel_xy_exp``, command range, ``feet_air_time``
   * - 발이 많이 미끄러진다
     - ``feet_slide``, foot collision, friction, contact sensor
   * - 양발이 동시에 공중에 뜨는 점프가 반복된다
     - ``no_jumps``, contact-force threshold/history, ``feet_air_time`` 의 single-stance 보상 크기
   * - 발날 또는 발끝으로 착지한다
     - ankle actuator gain/target, Foot body frame, contact sensor를 먼저 확인한다. ``feet_flat_orientation_l2`` 는 world-up 기준 함수이므로 평지 진단 실험에서만 선택적으로 활성화한다.
   * - 다리를 과도하게 벌리거나 교차한다
     - ``feet_lateral_separation_l2`` 의 signed width와 penalty를 먼저 확인한다. 이 term은 교차/최소 폭만 다루며 넓은 stance는 제한하지 않는다. 넓은 stance 문제에는 ``joint_deviation_hip`` 와 비활성 ``feet_stance_width_l2`` 를 별도로 검토한다.
   * - 관절이 떨린다
     - ``action_rate_l2``, ``second_action_rate_l2``, ``actuator_acc_l2``, actuator PD gain, action scale
   * - 토크가 과도하다
     - ``actuator_torques_l2``, actuator torque limit, action scale
   * - 관절이 limit 근처로 간다
     - ``dof_pos_limits``, ``joint_action_target_pos_limits``, 해당 joint의 default pose, action scale, USD hard limit
