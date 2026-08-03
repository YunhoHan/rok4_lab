RoK4 Observation Noise, Reset, Domain Randomization 문서
========================================================================

:작성일: 2026-07-30
:최종 업데이트: 2026-07-30
:대상 저장소: RoK4 repository root (``${ROK4_LAB_ROOT}``)
:기준 환경: Isaac Lab v2.3.2, Isaac Sim 5.1.0, ``env_isaaclab``

.. raw:: html

   <style>
   @media print {
     body, main {
       background-color: white;
     }
     pre.code {
       break-inside: avoid;
       page-break-inside: avoid;
     }
   }
   </style>

문서 경로 표기
--------------------------------------------

이 문서의 ``${ROK4_LAB_ROOT}`` 와 ``${ISAACLAB_ROOT}`` 는 특정 사용자의 홈 디렉터리가 아니라 각 저장소의
root를 뜻한다. 저장소를 설치한 실제 위치에 맞춰 한 번만 설정한다.

.. code-block:: bash

   export ROK4_LAB_ROOT="${HOME}/rok4_lab"
   export ISAACLAB_ROOT="${HOME}/IsaacLab"
   cd "${ISAACLAB_ROOT}"

다른 위치에 clone했다면 오른쪽 경로만 바꾸면 된다. 이후 명령은 다음처럼 사용자 이름과 무관하게 실행한다.

.. code-block:: bash

   ./isaaclab.sh -p "${ROK4_LAB_ROOT}/scripts/rsl_rl/train.py" \
     --task RoK4-Isaac-Velocity-Flat-v0 \
     --num_envs 4096 \
     --max_iterations 5000 \
     --headless

개요
--------------------------------------------

RoK4의 robustness 설정은 서로 목적과 적용 시점이 다른 세 층으로 나뉜다.

.. list-table::
   :header-rows: 1

   * - 구분
     - 바꾸는 대상
     - 표본화 시점
     - episode 중 변경
   * - Observation noise
     - policy가 받는 측정값
     - 매 policy observation 계산
     - 예
   * - Reset randomization
     - episode 시작 상태
     - 각 environment의 episode reset
     - reset 전까지 아니요
   * - Physics domain randomization
     - 접촉 재질, 질량, COM, joint 물성
     - 현재 대부분 scene startup
     - 아니요
   * - Training push
     - root XY velocity
     - 환경별 ``10~15 s`` interval
     - 예

Observation noise는 센서 오차를 흉내 내고, reset randomization은 다양한 초기 상태에서 회복하게 하며,
physics DR은 simulator와 다른 물성에서도 동작하는 policy를 유도한다. 세 항목은 서로 대체 관계가 아니다.

구현 위치
--------------------------------------------

.. list-table::
   :header-rows: 1

   * - 파일
     - 역할
   * - ``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/flat_env_cfg.py``
     - policy/critic observation term과 noise, Train/Play/Teleop 차이 정의
   * - ``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/domain_randomization_cfg.py``
     - 물성 및 reset 범위와 event mode 정의
   * - ``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/mdp/events.py``
     - correlated Foot material과 joint reset 표본화 구현
   * - ``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/agents/rsl_rl_ppo_cfg.py``
     - actor/critic running observation normalization 설정

Observation noise
--------------------------------------------

현재 1-frame policy observation은 48차원이고 5-frame term-major history를 flatten하여 actor에 240차원으로
전달한다. Noise는 각 observation term을 계산할 때 additive uniform 값으로 새로 표본화되고, 그 결과가 history에
들어간다.

.. list-table::
   :header-rows: 1

   * - Policy term
     - 차원/frame
     - 현재 noise
     - 단위
   * - ``base_ang_vel``
     - 3
     - ``Uniform(-0.2, 0.2)``
     - rad/s
   * - ``projected_gravity``
     - 3
     - ``Uniform(-0.05, 0.05)``
     - dimensionless
   * - ``velocity_commands``
     - 3
     - 없음
     - m/s, m/s, rad/s
   * - ``actuator_pos_rel``
     - 13
     - ``Uniform(-0.01, 0.01)``
     - rad
   * - ``actuator_vel_rel``
     - 13
     - ``Uniform(-1.5, 1.5)``
     - rad/s
   * - ``last_action``
     - 13
     - 없음
     - normalized raw action

Actor observation의 ``enable_corruption=True`` 때문에 위 noise는 Train에서 활성화된다. Critic은 같은 policy
observation 240개와 현재 simulator ``base_lin_vel_b`` 3개를 함께 사용한다. Privileged term은
``enable_corruption=False`` 이므로 base linear velocity 자체에는 noise를 추가하지 않는다.

Play와 Teleop은 ``self.observations.policy.enable_corruption=False`` 로 설정하므로 policy observation noise를
사용하지 않는다. 이는 checkpoint를 같은 simulator 조건에서 재현하고 동작을 눈으로 평가하기 위한 설정이다.

Noise와 normalization은 다르다
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additive observation noise는 입력값에 실제 perturbation을 더한다.

.. math::

   \tilde{o}_t = o_t + \epsilon_t,
   \qquad \epsilon_t \sim U(\epsilon_{min}, \epsilon_{max})

RSL-RL의 ``actor_obs_normalization=True`` 와 ``critic_obs_normalization=True`` 는 running mean과 variance로
network 입력의 통계적 scale을 조정한다. Noise를 제거하거나 물리 단위의 오차를 대신하지 않는다. 처리 순서는
개념적으로 다음과 같다.

.. code-block:: text

   simulator state
     -> observation term
     -> additive observation noise
     -> 5-frame history
     -> RSL-RL running normalization
     -> Actor/Critic

Baseline ``d949d40`` 이후 observation noise 범위는 변경하지 않았다. Actuator position noise를 ``0.02``, ``0.05``
또는 ``0.1 rad`` 로 올리는 실험은 transmission 오차와 센서 오차를 분리하기 어려워 현재 보류한다.

Reset randomization
--------------------------------------------

Reset event는 environment마다 episode가 끝나는 시점에 독립적으로 실행된다. 4096개 environment의 episode
progress가 서로 다르므로 모든 로봇이 동시에 reset되지 않는다.

Joint state reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reset_joints_by_position_scale_and_velocity`` 는 position과 velocity를 서로 다른 방식으로 표본화한다.

.. math::

   q_{0,e,j} &= \mathrm{clamp}\left(q_{default,e,j} U_{e,j}(0.9,1.1)\right) \\
   \dot{q}_{0,e,j} &= \mathrm{clamp}\left(U_{e,j}(-0.1,0.1)\right) \; \mathrm{rad/s}

각 environment ``e`` 와 joint ``j`` 에 독립적인 random sample을 사용한다. Default position이 0인 관절은
multiplicative position scale을 적용해도 0이다. Velocity는 default velocity에 scale을 곱하지 않고 절대값을
직접 표본화한다. Default velocity가 0이므로 기존 ``reset_joints_by_scale`` 방식으로는 velocity DR이 생기지
않기 때문이다. 최종 값은 soft joint position/velocity limit 안으로 clamp된다.

Root state reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Root 항목
     - 현재 범위
   * - position x/y
     - 각각 ``Uniform(-0.25, 0.25) m`` offset
   * - yaw
     - ``Uniform(-3.14, 3.14) rad``
   * - linear velocity x/y
     - 각각 ``Uniform(-0.1, 0.1) m/s``
   * - linear velocity z
     - 0
   * - angular velocity roll/pitch
     - 0
   * - angular velocity yaw
     - ``Uniform(-0.1, 0.1) rad/s``

Episode length는 ``20 s`` 다. Timeout 자체와 illegal-contact termination은 baseline ``d949d40`` 이후 변경하지
않았다. Reset randomization은 termination 조건이 아니라 termination 또는 timeout 뒤의 다음 초기 상태를
결정한다.

Physics domain randomization
--------------------------------------------

현재 physics DR은 large-vectorized training의 CPU 비용을 줄이기 위해 대부분 ``startup`` mode다. Scene을 만들
때 environment별로 표본화되며, 같은 training process 안에서는 episode가 reset되어도 해당 물성이 유지된다.

Contact material
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

먼저 모든 robot collision shape를 nominal static/dynamic friction ``0.8/0.6`` 으로 설정한다. 이후 각 training
environment가 64개 material bucket 중 하나를 선택하고, 그 bucket을 좌우 Foot의 모든 collision shape에 함께
적용한다.

.. math::

   \mu_s &\sim U(0.5,0.9) \\
   \mu_d &= 0.75\mu_s \in [0.375,0.675] \\
   e &\sim U(0.1,0.3)

``mu_dynamic <= mu_static`` 을 항상 만족하고, 한 environment의 좌우 발이 같은 material을 사용하므로 material
DR 자체가 좌우 gait 비대칭을 만들지 않는다. Foot 이외 robot shape는 nominal friction ``0.8/0.6`` 을 유지한다.

Joint physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab의 ``randomize_joint_parameters`` startup event가 nominal RoK4 joint property를 scale한다.

.. math::

   \mu_{s,e,j} &= \mu_{s,j}^{nominal} U_{s,e,j}(0.8,1.2) \\
   c_{v,e,j} &= c_{v,j}^{nominal} U_{v,e,j}(0.8,1.2) \\
   I_{a,e,j} &= I_{a,j}^{nominal} U_{a,e,j}(0.8,1.2)

Static friction, viscous friction, armature sample은 environment와 joint별로 서로 독립적이다. Nominal 값이 0인
property는 multiplicative scale을 적용해도 0으로 유지된다. 이 event는 PhysX joint property만 바꾸며
``ROK4_ACTUATOR_KP``, ``ROK4_ACTUATOR_KD``, ADAPT matrix와 action scale은 변경하지 않는다.

Mass와 COM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Body group
     - Mass scale
     - COM offset range
   * - Base
     - ``Uniform(0.9,1.1)``
     - x/y/z 각각 ``+-0.01 m``
   * - Upper body
     - ``Uniform(0.9,1.25)``
     - x/y/z 각각 ``+-0.03 m``
   * - Lower body와 feet
     - ``Uniform(0.9,1.25)``
     - x/y/z 각각 ``+-0.005 m``

Training push와 external wrench
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

상속한 ``push_robot`` interval event는 environment별 독립 timer를 사용한다. ``10~15 s`` 후 root world-frame
XY velocity를 각각 ``Uniform(-0.5,0.5) m/s`` 로 바꾸고 다음 interval을 다시 표본화한다. 이 event는 지속적인
force가 아니라 순간적인 root velocity disturbance다.

``base_external_force_torque`` reset event는 남아 있지만 현재 force와 torque 범위가 모두 0이므로 실제 외력을
추가하지 않는다.

Train, Play, Teleop 차이
--------------------------------------------

.. list-table::
   :header-rows: 1

   * - 항목
     - Train
     - Play/Teleop 현재 코드
   * - Policy observation noise
     - 활성
     - 비활성
   * - Foot friction
     - static ``0.5~0.9``, dynamic ``0.75 * static``
     - nominal ``0.8/0.6``
   * - Foot restitution
     - ``0.1~0.3``
     - 현재 material event에 의해 ``0.1~0.3`` 유지
   * - Joint physics DR
     - 활성
     - ``joint_physics=None`` 으로 비활성
   * - Mass/COM startup DR
     - 활성
     - 현재 상속되어 활성
   * - Joint/root reset DR
     - 활성
     - 현재 상속되어 활성
   * - Automatic push
     - 활성
     - 비활성, UI 수동 push만 사용
   * - External reset wrench
     - 범위 0
     - event 제거

따라서 현재 Play/Teleop은 observation noise, Foot friction, joint physics, automatic push에 대해서는 평가용으로
단순화되어 있지만 mass/COM, reset state, Foot restitution까지 완전히 deterministic한 구성은 아니다. 완전한
deterministic evaluation이 필요하면 이 표의 남은 startup/reset event를 별도 실험에서 명시적으로 제거해야 한다.

시간축과 비동기 동작
--------------------------------------------

Episode timer, velocity-command freeze timer, training-push timer는 서로 다른 상태다.

.. list-table::
   :header-rows: 1

   * - Timer
     - 기준
     - Reset 시 동작
   * - Episode
     - environment별 progress, 최대 ``20 s``
     - 0으로 돌아감
   * - Periodic freeze
     - command term의 환경별 phase
     - 환경별 새 phase/duration 표본화
   * - Training push
     - event manager의 환경별 ``10~15 s`` time-left
     - 다음 interval 표본화

따라서 어떤 environment는 걷는 중에 push를 받고, 다른 environment는 standing 전환 직전이나 직후에 push를
받을 수 있다. 이 비동기성은 같은 학습 step에 모든 environment가 같은 사건을 겪는 것을 막는다.

현재 보류한 실험
--------------------------------------------

.. list-table::
   :header-rows: 1

   * - 항목
     - 현재 상태
     - 보류 이유
   * - Gravity DR
     - 없음
     - 기본 physics/reset DR 효과를 먼저 분리해 확인
   * - Joint-limit DR
     - 없음
     - action feasibility와 reward 해석이 함께 바뀜
   * - ADAPT transmission ratio/topology DR
     - 없음
     - actuator state/action/torque mapping이 동시에 바뀌므로 별도 검증 필요
   * - Larger actuator-position observation noise
     - 없음
     - 센서 오차와 transmission model mismatch를 구분한 뒤 조정

권장 실험 순서
--------------------------------------------

한 번에 여러 범위를 바꾸면 gait 변화의 원인을 찾기 어렵다. 현재 baseline을 보존하고 다음 순서로 하나씩 비교한다.

.. code-block:: text

   1. Current noise + current reset + current physics DR
   2. Sim2Sim에서 동일 checkpoint의 물성 민감도 확인
   3. Reset/recovery 범위 조정
   4. Contact와 joint physics DR 범위 조정
   5. 실제 sensor 특성에 맞춘 observation noise 조정
   6. 마지막으로 transmission 또는 gravity/joint-limit DR 검토

각 실험은 config diff, random seed, checkpoint, command 범위와 Play/Teleop 조건을 함께 기록해야 비교가 가능하다.
