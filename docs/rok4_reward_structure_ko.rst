RoK4 Reward Structure
=====================

작성일: 2026-07-07

이 문서는 ``RoK4-Isaac-Velocity-Flat-v0`` task의 현재 reward 구조와 reward function 설정을 정리한다.
현재 reward는 Isaac Lab G1 velocity task 구조를 RoK4 링크/관절 이름에 맞게 이식한 flat walking 초기
baseline이다. 최종 튜닝값이 아니라 첫 보행 실험을 위한 시작점으로 봐야 한다.

관련 파일
---------

.. list-table::
   :header-rows: 1

   * - 역할
     - 파일
   * - RoK4 flat task 설정
     - ``/home/rclab/rok4_lab/source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/flat_env_cfg.py``
   * - 공통 locomotion reward term 기본값
     - ``/home/rclab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py``
   * - locomotion 전용 reward 함수
     - ``/home/rclab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py``
   * - Isaac Lab 공통 reward 함수
     - ``/home/rclab/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py``
   * - mdp namespace 연결
     - ``/home/rclab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py``

구조 요약
---------

``RoK4FlatEnvCfg``는 환경 설정이고, ``RoK4RewardsCfg``는 reward term 묶음이다.

.. code-block:: text

   RoK4FlatEnvCfg
     ├─ 상속: LocomotionVelocityRoughEnvCfg
     └─ 포함: rewards = RoK4RewardsCfg()

   RoK4RewardsCfg
     └─ 상속: RewardsCfg

   각 reward term
     └─ RewardTermCfg(func=mdp.xxx, weight=..., params=...)

   mdp.xxx
     ├─ locomotion 전용 함수: velocity/mdp/rewards.py
     └─ Isaac Lab 공통 함수: isaaclab/envs/mdp/rewards.py

즉 ``RoK4FlatEnvCfg -> RoK4RewardsCfg``는 상속이 아니라 포함/사용 관계이고,
``RoK4RewardsCfg -> RewardsCfg``는 상속 관계이다. ``mdp/rewards.py``는 부모 클래스가 아니라 실제 reward
계산 함수가 들어 있는 함수 모음이다.

실제로는 다음처럼 이해하면 된다.

.. code-block:: text

   RoK4RewardsCfg는 RewardsCfg를 상속받아서 기본 reward term들을 물려받는다.
   RoK4에 필요한 reward는 mdp.xxx 함수를 직접 지정해 새로 정의하거나 override한다.
   RewardTermCfg/RewTerm은 mdp.xxx 함수, weight, params를 하나의 reward term으로 묶는다.

Reward 계산 방식
----------------

Isaac Lab의 ``RewardManager``는 각 reward term의 raw value를 계산한 뒤 ``weight``를 곱하고, 모든 term을
합산한다.

.. code-block:: text

   total_reward =
       sum(weight_i * reward_function_i(env, params_i))

따라서 positive weight는 행동을 유도하고, negative weight는 penalty로 작동한다.

RoK4 전용 Reward Terms
---------------------

아래 term들은 ``RoK4RewardsCfg``에서 직접 정의하거나 부모 ``RewardsCfg``의 term을 RoK4용으로 override한
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
   * - ``feet_air_time``
     - ``mdp.feet_air_time_positive_biped``
     - ``0.75``
     - feet: ``L_Foot_Link``, ``R_Foot_Link``; ``threshold=0.4``
     - 양발 보행에서 한 발씩 공중에 있는 step pattern을 유도한다.
   * - ``feet_slide``
     - ``mdp.feet_slide``
     - ``-0.2``
     - feet: ``L_Foot_Link``, ``R_Foot_Link``
     - 지면 접촉 중인 발이 미끄러지는 것을 줄인다.
   * - ``dof_pos_limits``
     - ``mdp.joint_pos_limits``
     - ``-1.0``
     - ``.*_Ankle_Pitch_Joint``, ``.*_Ankle_Roll_Joint``
     - ankle joint가 soft joint limit을 넘는 것을 강하게 막는다.
   * - ``joint_deviation_hip``
     - ``mdp.joint_deviation_l1``
     - ``-0.1``
     - ``.*_Hip_Yaw_Joint``, ``.*_Hip_Roll_Joint``
     - hip yaw/roll이 default pose에서 과도하게 벗어나지 않게 한다.
   * - ``joint_deviation_torso``
     - ``mdp.joint_deviation_l1``
     - ``-0.1``
     - ``Torso_Yaw_Joint``
     - torso yaw가 default pose에서 과도하게 벗어나지 않게 한다.

부모 RewardsCfg에서 상속받고 RoK4에서 수정한 Terms
-------------------------------------------------

아래 term들은 ``RewardsCfg``에서 기본으로 제공되며, ``RoK4FlatEnvCfg.__post_init__()``에서 weight나 body/joint
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
     - ``-1.0``
     - 기본 robot root
     - 몸이 기울어지는 것을 줄이고 upright 자세를 유도한다.
   * - ``action_rate_l2``
     - ``mdp.action_rate_l2``
     - ``-0.005``
     - action 전체
     - action이 step마다 급격히 변하는 것을 줄인다.
   * - ``dof_acc_l2``
     - ``mdp.joint_acc_l2``
     - ``-1.0e-7``
     - ``.*_Hip_.*``, ``.*_Knee_Pitch_Joint``
     - hip/knee joint acceleration을 줄여 움직임을 부드럽게 한다.
   * - ``dof_torques_l2``
     - ``mdp.joint_torques_l2``
     - ``-2.0e-6``
     - ``.*_Hip_.*``, ``.*_Knee_Pitch_Joint``, ``.*_Ankle_.*``
     - hip/knee/ankle torque 사용량을 줄인다.
   * - ``undesired_contacts``
     - ``mdp.undesired_contacts``
     - ``-1.0``
     - ``Base_Link``, ``Upper_Body_Link``; ``threshold=1.0``
     - 발이 아닌 base/upper body 접촉을 penalty로 처리한다.

Reward Function 요약
--------------------

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
     - yaw frame x/y 선속도 command error에 ``exp(-error / std^2)``를 적용한다.
   * - ``track_ang_vel_z_world_exp``
     - locomotion mdp
     - world z축 angular velocity command error에 ``exp(-error / std^2)``를 적용한다.
   * - ``feet_air_time_positive_biped``
     - locomotion mdp
     - 한 발 지지 상태의 air/contact time을 보상하며, command 크기가 작으면 0이 된다.
   * - ``feet_slide``
     - locomotion mdp
     - 접촉 중인 foot의 horizontal linear velocity norm을 penalty로 계산한다.
   * - ``lin_vel_z_l2``
     - Isaac Lab 공통
     - base body-frame z velocity squared.
   * - ``ang_vel_xy_l2``
     - Isaac Lab 공통
     - base body-frame roll/pitch angular velocity squared sum.
   * - ``flat_orientation_l2``
     - Isaac Lab 공통
     - projected gravity의 x/y component squared sum으로 몸 기울기를 penalty화한다.
   * - ``joint_torques_l2``
     - Isaac Lab 공통
     - 선택된 joint의 applied torque squared sum.
   * - ``joint_acc_l2``
     - Isaac Lab 공통
     - 선택된 joint의 joint acceleration squared sum.
   * - ``action_rate_l2``
     - Isaac Lab 공통
     - 현재 action과 이전 action 차이의 squared sum.
   * - ``joint_deviation_l1``
     - Isaac Lab 공통
     - 현재 joint position과 default joint position 차이의 absolute sum.
   * - ``joint_pos_limits``
     - Isaac Lab 공통
     - soft joint position limit을 넘은 정도를 합산한다.
   * - ``undesired_contacts``
     - Isaac Lab 공통
     - 지정 body의 contact force가 threshold를 넘은 contact 개수를 penalty로 반환한다.

Command와 Reward의 연결
----------------------

현재 RoK4 flat command range는 다음과 같다.

.. code-block:: python

   lin_vel_x = (0.0, 0.8)
   lin_vel_y = (-0.3, 0.3)
   ang_vel_z = (-0.6, 0.6)

이 command는 ``track_lin_vel_xy_exp``, ``track_ang_vel_z_exp``, ``feet_air_time``에 직접 영향을 준다. 특히
``feet_air_time_positive_biped``는 x/y command norm이 ``0.1`` 이하이면 reward를 0으로 만든다. 즉 거의 정지
명령에서는 stepping reward가 강하게 작동하지 않는다.

Termination과 Reward의 연결
---------------------------

RoK4 flat task는 부모 ``TerminationsCfg``의 ``base_contact``를 사용하되 body names를 RoK4에 맞게 바꾼다.

.. code-block:: python

   self.terminations.base_contact.params["sensor_cfg"].body_names = ["Base_Link", "Upper_Body_Link"]

따라서 ``Base_Link`` 또는 ``Upper_Body_Link``가 지면과 강하게 접촉하면 episode termination이 발생하고,
``termination_penalty = -200.0``가 함께 작동한다. 학습 초반에 너무 자주 종료되면 이 body list 또는 threshold,
초기 자세, command range를 함께 확인해야 한다.

현재 설계 의도
--------------

현재 reward는 다음 목표를 동시에 가진다.

.. code-block:: text

   1. command velocity를 따라간다.
   2. upright 자세를 유지한다.
   3. 한 발씩 드는 biped stepping pattern을 만든다.
   4. 발 미끄러짐을 줄인다.
   5. torque, joint acceleration, action rate를 줄여 움직임을 부드럽게 한다.
   6. ankle limit, hip/torso deviation을 제한한다.
   7. base/upper body 접촉을 넘어짐으로 처리한다.

튜닝 시 우선 확인할 항목
-----------------------

.. list-table::
   :header-rows: 1

   * - 현상
     - 먼저 볼 항목
   * - 너무 자주 넘어진다
     - ``termination_penalty``, ``base_contact``, ``flat_orientation_l2``, 초기 자세
   * - 거의 걷지 않고 버틴다
     - ``track_lin_vel_xy_exp``, command range, ``feet_air_time``
   * - 발이 많이 미끄러진다
     - ``feet_slide``, foot collision, friction, contact sensor
   * - 관절이 떨린다
     - ``action_rate_l2``, ``dof_acc_l2``, PD gain, action scale
   * - 토크가 과도하다
     - ``dof_torques_l2``, effort limit, action scale
   * - 발목이 limit 근처로 간다
     - ``dof_pos_limits``, ankle default pose, action scale
