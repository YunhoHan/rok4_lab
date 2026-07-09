RoK4 Flat RSL-RL Task 구조 문서
================================

:작성일: 2026-07-09
:대상 저장소: ``/home/rclab/rok4_lab``
:기준 환경: Isaac Lab v2.3.2, Isaac Sim 5.1.0, ``env_isaaclab``

개요
----

이번 작업의 목적은 RoK4를 위한 첫 강화학습 환경을 만드는 것이다. 바로 rough terrain으로 가지 않고,
먼저 flat terrain에서 blind velocity walking을 학습할 수 있는 최소 구조를 만들었다.

중요한 점은 G1을 그대로 복사한 것이 아니라는 점이다. G1은 잘 정리된 Isaac Lab humanoid locomotion
task 구조를 참고한 템플릿이고, 실제 로봇 asset, 관절 순서, action dimension, reward body 이름은 모두
RoK4 기준으로 다시 작성했다.

현재 생성된 task는 다음 두 개다.

.. list-table::
   :header-rows: 1

   * - Task 이름
     - 목적
   * - ``RoK4-Isaac-Velocity-Flat-v0``
     - ``rok4_train.usd``를 사용하는 RoK4 flat-ground velocity tracking 학습용 task
   * - ``RoK4-Isaac-Velocity-Flat-Play-v0``
     - visual mesh가 있는 ``rok4_test.usd``로 학습된 RoK4 flat policy를 재생하는 task

전체 구조
---------

현재 RoK4 Lab 저장소의 핵심 구조는 다음과 같다.

.. code-block:: text

   rok4_lab/
     assets/
       rok4_wholebody/                  # URDF/USD/STL asset bundle, git ignore
     source/
       rok4_tasks/
         rok4_tasks/
           __init__.py                  # RoK4 task registration import
           assets/
             robots/
               rok4.py                  # RoK4 asset, actuator, joint order 정의
           manager_based/
             locomotion/
               velocity/
                 config/
                   rok4/
                     __init__.py        # Gym task 등록
                     flat_env_cfg.py    # RoK4 Flat env/reward/action/obs 정의
                     agents/
                       rsl_rl_ppo_cfg.py # RSL-RL PPO 설정
     scripts/
       check_rok4_zero.py               # 모델/PD zero hold 검사
       check_rok4_random.py             # sinusoidal command 검사
       check_rok4_joint_monkey.py       # joint axis/limit 검사
       rsl_rl/
         _run_isaaclab_rsl.py           # Isaac Lab 원본 train/play wrapper
         train.py                       # RoK4 task 등록 후 Isaac Lab train 실행
         play.py                        # RoK4 task 등록 후 Isaac Lab play 실행

RoK4 구조 관계
--------------

현재 RoK4 flat task는 ``flat_env_cfg.py``를 중심으로 움직인다. 다만 실행 entry point는
``config/rok4/__init__.py``의 Gym task 등록이고, ``flat_env_cfg.py``는 환경 설정의 main config 역할을 한다.

.. code-block:: text

   [사용자 실행]
   ./isaaclab.sh -p /home/rclab/rok4_lab/scripts/rsl_rl/train.py
       |
       v
   scripts/rsl_rl/train.py
       |
       v
   scripts/rsl_rl/_run_isaaclab_rsl.py
       |  RoK4 package path 추가
       |  Isaac Lab 원본 rsl_rl/train.py 실행 전 import rok4_tasks 삽입
       v
   source/rok4_tasks/rok4_tasks/__init__.py
       |
       v
   manager_based/locomotion/velocity/config/rok4/__init__.py
       |  Gym task 등록
       |  id = RoK4-Isaac-Velocity-Flat-v0
       |
       +--> env_cfg_entry_point
       |      flat_env_cfg.py : RoK4FlatEnvCfg
       |
       +--> rsl_rl_cfg_entry_point
              agents/rsl_rl_ppo_cfg.py : RoK4FlatPPORunnerCfg

환경 설정 관계는 다음과 같다.

.. code-block:: text

   RoK4FlatEnvCfg
     ├─ 상속: IsaacLab LocomotionVelocityRoughEnvCfg
     ├─ 포함: rewards = RoK4RewardsCfg()
     ├─ 학습 사용: ROK4_TRAIN_CFG
     ├─ 사용: ROK4_JOINT_ORDER
     └─ 사용: ROK4_ACTION_SCALE

   RoK4FlatEnvCfg_PLAY
     ├─ 상속: RoK4FlatEnvCfg
     └─ 재생 사용: ROK4_TEST_CFG

   RoK4RewardsCfg
     └─ 상속: IsaacLab RewardsCfg

   RoK4FlatPPORunnerCfg
     └─ RSL-RL PPO runner/network/algorithm 설정

   RewardTermCfg(func=mdp.xxx)
     └─ IsaacLab mdp reward 함수 호출

각 파일의 관계를 역할로 나누면 다음과 같다.

.. list-table::
   :header-rows: 1

   * - 파일
     - 관계
     - 역할
   * - ``config/rok4/__init__.py``
     - task 등록
     - ``RoK4-Isaac-Velocity-Flat-v0`` 이름을 ``RoK4FlatEnvCfg``와 ``RoK4FlatPPORunnerCfg``에 연결
   * - ``flat_env_cfg.py``
     - 환경 main config
     - RoK4 scene, terrain, action, observation, event, reward, command, termination 설정
   * - ``agents/rsl_rl_ppo_cfg.py``
     - 학습 config
     - RSL-RL PPO network와 algorithm hyperparameter 설정
   * - ``assets/robots/rok4.py``
     - robot asset config
     - USD path, initial pose, actuator, joint order, action scale 제공
   * - IsaacLab ``velocity_env_cfg.py``
     - 부모 config
     - manager-based velocity locomotion의 공통 scene/action/obs/reward/termination 틀 제공
   * - IsaacLab ``mdp/rewards.py``
     - reward 함수 모음
     - ``RewardTermCfg(func=mdp.xxx)``가 호출하는 실제 reward 계산 함수 제공

한 줄로 요약하면, ``flat_env_cfg.py``는 RoK4 환경의 중심 설정 파일이고, ``__init__.py``는 task 이름을
등록하는 입구, ``rsl_rl_ppo_cfg.py``는 학습기 설정, ``rok4.py``는 로봇 asset 설정이다.

새로 생성된 파일
----------------

``source/rok4_tasks/rok4_tasks/manager_based/__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manager-based task 패키지 시작점이다. Isaac Lab의 ``ManagerBasedRLEnv`` 스타일 task를 RoK4 저장소 안에
넣기 위한 디렉터리 계층이다.

``source/rok4_tasks/rok4_tasks/manager_based/locomotion/__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Locomotion task 그룹 시작점이다. 향후 velocity walking 외에 balancing, standing, whole-body tracking 같은
locomotion 계열 task를 추가할 때 같은 계층 아래에 둘 수 있다.

``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Velocity tracking task 그룹 시작점이다. 현재 RoK4의 첫 학습 목표가 command velocity를 따라 걷는 것이므로
이 계층을 만들었다.

``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Velocity task 안에서 robot별 config를 묶기 위한 계층이다. 지금은 ``rok4``만 있지만, 나중에 RoK4 variant가
늘어나면 같은 구조 아래에 추가할 수 있다.

``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gymnasium task를 등록하는 파일이다. 여기에서 다음 task 이름이 Isaac Lab registry에 올라간다.

.. code-block:: text

   RoK4-Isaac-Velocity-Flat-v0
   RoK4-Isaac-Velocity-Flat-Play-v0

각 task는 ``isaaclab.envs:ManagerBasedRLEnv``를 entry point로 사용한다. 환경 설정은
``flat_env_cfg.py``에서, RSL-RL PPO 설정은 ``agents/rsl_rl_ppo_cfg.py``에서 불러온다.

``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/flat_env_cfg.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

이번 작업의 핵심 파일이다. RoK4 flat walking 환경을 정의한다.

주요 역할은 다음과 같다.

.. list-table::
   :header-rows: 1

   * - 항목
     - 내용
   * - Scene
     - 학습 task는 ``ROK4_TRAIN_CFG``를 ``{ENV_REGEX_NS}/Robot`` 위치에 spawn
   * - Terrain
     - plane terrain 사용, rough terrain generator와 height scanner 제거
   * - Action
     - ``ROK4_JOINT_ORDER`` 기준 13차원 joint position target
   * - Observation
     - blind proprioceptive history observation
   * - Reward
     - velocity tracking, upright, action smoothness, torque/acc penalty, foot contact 관련 reward
   * - Termination
     - timeout, base/upper-body illegal contact
   * - Play cfg
     - ``ROK4_TEST_CFG``로 교체, 적은 env 수, corruption off, push/random external force off

RoK4 action space는 다음 13축이다.

.. code-block:: text

   Left leg:
     L_Hip_Yaw_Joint
     L_Hip_Roll_Joint
     L_Hip_Pitch_Joint
     L_Knee_Pitch_Joint
     L_Ankle_Pitch_Joint
     L_Ankle_Roll_Joint

   Right leg:
     R_Hip_Yaw_Joint
     R_Hip_Roll_Joint
     R_Hip_Pitch_Joint
     R_Knee_Pitch_Joint
     R_Ankle_Pitch_Joint
     R_Ankle_Roll_Joint

   Torso:
     Torso_Yaw_Joint

관절 순서는 반드시 ``rok4.py``의 ``ROK4_JOINT_ORDER``를 따른다. G1의 관절 순서를 그대로 쓰지 않는다.

Actor observation은 5-step history를 사용한다.

.. code-block:: text

   base_ang_vel
   projected_gravity
   velocity_commands
   joint_pos
   joint_vel
   last_action

Actor에는 다음 정보를 넣지 않는다.

.. code-block:: text

   camera image
   terrain height scan
   base linear velocity

따라서 현재 구조는 estimator가 없는 history MLP 방식의 blind walking baseline이다.

RoK4 control timing
~~~~~~~~~~~~~~~~~~~

RoK4는 Isaac Lab 본체의 ``LocomotionVelocityRoughEnvCfg``를 수정하지 않고, ``RoK4FlatEnvCfg`` 안에서
simulation/control timing만 override한다.

.. list-table::
   :header-rows: 1

   * - 설정
     - 값
   * - ``sim.dt``
     - ``0.002 s``
   * - physics frequency
     - ``500 Hz``
   * - ``decimation``
     - ``5``
   * - policy/action period
     - ``0.010 s``
   * - policy/action frequency
     - ``100 Hz``

``source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/agents/rsl_rl_ppo_cfg.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RSL-RL PPO runner 설정 파일이다.

현재 설정은 RoK4용 network/observation normalization 설정에 G1-style PPO algorithm 값을 섞은 flat walking
초기 baseline이다. 이 값들은 최종 튜닝값이 아니라 첫 RoK4 flat 보행 실험을 위한 시작점이다.

.. list-table::
   :header-rows: 1

   * - 설정
     - 값
   * - experiment name
     - ``rok4_flat``
   * - max iterations
     - ``5000``
   * - steps per env
     - ``24``
   * - actor hidden dims
     - ``[512, 256, 128]``
   * - critic hidden dims
     - ``[512, 256, 128]``
   * - actor/critic obs normalization
     - ``True``
   * - activation
     - ``elu``
   * - learning rate
     - ``1.0e-3``
   * - entropy coef
     - ``0.008``
   * - value loss coef
     - ``1.0``
   * - desired KL
     - ``0.01``
   * - obs groups
     - actor와 critic 모두 ``policy`` observation 사용

첫 버전에서는 asymmetric critic이나 estimator를 넣지 않았다. Flat walking이 먼저 돌아가는지 확인한 뒤,
rough terrain이나 estimator를 단계적으로 추가하기 위함이다.

``scripts/rsl_rl/_run_isaaclab_rsl.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab 원본 ``scripts/reinforcement_learning/rsl_rl/train.py``와 ``play.py``를 수정하지 않고 RoK4 task를
등록하기 위한 wrapper이다.

동작 흐름은 다음과 같다.

.. code-block:: text

   1. rok4_lab/source/rok4_tasks 를 sys.path에 추가
   2. 현재 작업 디렉터리의 Isaac Lab RSL-RL script 경로 찾기
   3. Isaac Lab 원본 train.py/play.py source를 읽기
   4. 원본의 import isaaclab_tasks 바로 뒤에 import rok4_tasks 삽입
   5. 원본 script를 실행

이 방식의 장점은 Isaac Lab 본체를 수정하지 않는다는 점이다.

``scripts/rsl_rl/train.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~

RoK4 task를 등록한 뒤 Isaac Lab 원본 RSL-RL training script를 실행한다.

사용자는 Isaac Lab root에서 다음처럼 실행한다.

.. code-block:: bash

   ./isaaclab.sh -p /home/rclab/rok4_lab/scripts/rsl_rl/train.py \
     --task RoK4-Isaac-Velocity-Flat-v0 \
     --num_envs 512 \
     --max_iterations 5000 \
     --headless

``scripts/rsl_rl/play.py``
~~~~~~~~~~~~~~~~~~~~~~~~~

RoK4 task를 등록한 뒤 Isaac Lab 원본 RSL-RL playback script를 실행한다.

사용자는 Isaac Lab root에서 다음처럼 실행한다.

.. code-block:: bash

   ./isaaclab.sh -p /home/rclab/rok4_lab/scripts/rsl_rl/play.py \
     --task RoK4-Isaac-Velocity-Flat-Play-v0 \
     --num_envs 16 \
     --checkpoint /path/to/model.pt

수정된 기존 파일
----------------

``source/rok4_tasks/rok4_tasks/__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

기존에는 패키지 docstring만 있었다. 이제 RoK4 task registry가 import되도록 다음 역할을 한다.

.. code-block:: python

   from .manager_based.locomotion.velocity.config import rok4

이 import가 실행되어야 ``RoK4-Isaac-Velocity-Flat-v0`` task가 gym registry에 등록된다.

``README.md``
~~~~~~~~~~~~~

새 Flat RL Task 섹션을 추가했다. task 이름, observation/action 구조, smoke test, normal training,
play 명령어를 문서화했다.

``CHANGELOG.md``
~~~~~~~~~~~~~~~~

``Unreleased`` 항목에 RoK4 flat RSL-RL task 추가 내용을 기록했다.

학습 실행 흐름
--------------

학습 명령을 실행하면 전체 흐름은 다음과 같다.

.. code-block:: text

   ./isaaclab.sh
     -> /home/rclab/rok4_lab/scripts/rsl_rl/train.py
       -> _run_isaaclab_rsl.py
         -> sys.path에 rok4_tasks 추가
         -> Isaac Lab 원본 train.py 실행
           -> import isaaclab_tasks
           -> import rok4_tasks
             -> RoK4 task gym 등록
           -> hydra_task_config가 env_cfg / agent_cfg 로드
           -> gym.make("RoK4-Isaac-Velocity-Flat-v0")
           -> ManagerBasedRLEnv 생성
           -> RslRlVecEnvWrapper
           -> OnPolicyRunner
           -> PPO 학습

스모크 테스트 결과
------------------

다음 smoke test를 수행했다.

.. code-block:: bash

   ./isaaclab.sh -p /home/rclab/rok4_lab/scripts/rsl_rl/train.py \
     --task RoK4-Isaac-Velocity-Flat-v0 \
     --num_envs 2 \
     --max_iterations 1 \
     --headless

확인된 결과는 다음과 같다.

.. list-table::
   :header-rows: 1

   * - 항목
     - 결과
   * - 환경 생성
     - 성공
   * - action shape
     - ``13``
   * - observation shape
     - ``240``
   * - actor input
     - 5-step history 기반 240차원
   * - actor output
     - 13차원 joint position target
   * - PPO 1 iteration
     - 성공
   * - 생성 checkpoint
     - ``/home/rclab/IsaacLab/logs/rsl_rl/rok4_flat/2026-07-07_13-44-26/model_0.pt``

현재 설계 의도
--------------

현재 task는 최종 rough terrain 정책이 아니다. 첫 번째 목표는 원인 분리가 쉬운 flat walking baseline이다.

현재 단계에서 의도적으로 넣지 않은 항목은 다음과 같다.

.. code-block:: text

   rough terrain
   terrain curriculum
   height scanner
   camera observation
   explicit estimator / MLP encoder
   teacher-student distillation
   asymmetric privileged critic

추후 권장 순서는 다음과 같다.

.. code-block:: text

   1. Flat smoke test 확인
   2. Flat 학습으로 서기/걷기 안정화
   3. reward, action scale, termination 조정
   4. RoK4 Rough task 추가
   5. weak rough terrain curriculum
   6. estimator 또는 teacher-student 구조 추가

주요 실행 명령어
----------------

Smoke test:

.. code-block:: bash

   cd /home/rclab/rok4_lab
   ROK4LAB_DIR=$(pwd)

   cd /home/rclab/IsaacLab
   conda activate env_isaaclab

   ./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/train.py \
     --task RoK4-Isaac-Velocity-Flat-v0 \
     --num_envs 2 \
     --max_iterations 1 \
     --headless

Normal training:

.. code-block:: bash

   ./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/train.py \
     --task RoK4-Isaac-Velocity-Flat-v0 \
     --num_envs 512 \
     --max_iterations 5000 \
     --headless

Play:

.. code-block:: bash

   ./isaaclab.sh -p ${ROK4LAB_DIR}/scripts/rsl_rl/play.py \
     --task RoK4-Isaac-Velocity-Flat-Play-v0 \
     --num_envs 16 \
     --checkpoint /path/to/model.pt
