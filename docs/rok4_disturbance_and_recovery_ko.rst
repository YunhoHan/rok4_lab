RoK4 혼합 외란과 Recovery 학습 문서
===============================================

:작성 기준: 2026-08-26
:적용 브랜치: ``yunho/mixed-push-disturbance``
:기준 브랜치: ``yunho/concurrent-state-estimator``
:대상 task: ``RoK4-Isaac-Velocity-Flat-v0``

문서 경로
-----------------------------------------------

이 문서에서 ``${ROK4_LAB_ROOT}`` 는 RoK4 저장소, ``${ISAACLAB_ROOT}`` 는 Isaac Lab 저장소의 root를 뜻한다.
개인 PC의 ``/home/<user>`` 경로를 복사하지 말고 실제 clone 위치에 맞춰 사용한다.

목적
-----------------------------------------------

기준 정책은 방향별 보행, 양발 정지, 약한 급정지와 순간 velocity push에 대한 복원을 이미 학습했다. 그러나 기존
자동 외란은 world-frame root velocity를 순간적으로 바꾸는 한 종류뿐이었다. 이 구조는 post-disturbance state를
빠르게 만들지만 실제 사람이 일정 시간 미는 상황, 접촉 중 policy가 반응하는 과정, RoK4 발의 전후 지지영역
비대칭을 충분히 표현하지 못한다.

이번 실험은 command, freeze, reward, estimator와 observation을 그대로 유지하고 자동 push만 다음과 같이 바꾼다.

* 외란 방향을 world XY가 아니라 **현재 robot base-yaw XY** 에서 정의한다.
* 한 event마다 같은 ``Delta v_b`` 후보를 표본화한다.
* 50%는 기존과 같은 순간 velocity push, 50%는 impulse-equivalent force pulse로 적용한다.
* 두 mode를 동시에 적용하지 않으므로 event 빈도와 목표 운동량을 중복시키지 않는다.
* RoK4의 전방 지지영역이 후방보다 길다는 관찰을 반영해 longitudinal 범위를 비대칭으로 둔다.

좌표계와 방향
-----------------------------------------------

외란 표본은 base roll/pitch를 제거한 yaw frame에서 정의한다.

.. math::

   \Delta\mathbf v_b =
   [\Delta v_{x,b},\ \Delta v_{y,b},\ 0]^{T}

.. math::

   \Delta v_{x,b}\sim U(-0.5,1.0),\qquad
   \Delta v_{y,b}\sim U(-0.5,0.5)\quad [\mathrm{m/s}]

``+X`` 는 로봇의 현재 전방, ``-X`` 는 후방, ``+Y`` 는 좌측, ``-Y`` 는 우측이다. Event가 시작하는 순간의
root quaternion에서 yaw만 취해 world frame으로 회전한다.

.. math::

   \Delta\mathbf v_w =
   R_{wb,\mathrm{yaw}}\Delta\mathbf v_b

따라서 로봇이 world에서 어느 방향을 보고 있더라도 외란의 전후좌우 의미가 유지된다. Force mode도 시작 순간
world 방향을 계산한 뒤 pulse가 끝날 때까지 그 world 방향을 고정한다. 로봇이 pulse 도중 회전한다고 힘이 robot
frame을 따라 회전하지 않는다.

전후 비대칭의 근거와 주의점
-----------------------------------------------

Linear inverted pendulum 근사에서 DCM은 다음과 같다.

.. math::

   \boldsymbol\xi = \mathbf x + \frac{\mathbf v}{\omega_0},
   \qquad \omega_0=\sqrt{\frac{g}{h}}

순간 속도 변화가 만드는 DCM 이동은

.. math::

   \Delta\boldsymbol\xi = \frac{\Delta\mathbf v}{\omega_0}

이다. RoK4 기본 높이 ``h=0.907 m`` 를 쓰면 ``omega_0`` 는 약 ``3.29 s^-1`` 이다. 발이 ankle보다 전방으로
더 길면 전방 DCM margin도 후방보다 크므로 같은 크기의 외란에서 전방은 오래 버티고 후방은 먼저 stepping할 수
있다. 이번 ``[-0.5,1.0]`` 범위는 더 큰 전방 외란으로 이 버티기 전략을 넘어 recovery step을 경험시키려는 첫
실험값이다.

중요하게도 이 범위는 zero-mean이 아니다.

.. math::

   E[\Delta v_{x,b}] = \frac{-0.5+1.0}{2}=+0.25\ \mathrm{m/s}

즉 전방 외란을 의도적으로 더 강하게 노출한다. 학습 결과가 forward gait 또는 standing COM 자체를 전방으로
편향시키면, 다음 실험에서는 실제 foot support margin을 측정하고 부호 확률 또는 conditional magnitude를 조정해
평균 impulse를 0에 맞춰야 한다. 이 branch의 값은 support polygon을 정밀 측정한 최종값이 아니라 recovery
동작을 확인하는 실험값이다.

Mode 1: 순간 velocity push
-----------------------------------------------

확률 ``0.5`` 로 현재 world-frame root velocity에 회전된 ``Delta v_w`` 를 더한다.

.. math::

   \mathbf v_{w,\mathrm{new}}
   =\mathbf v_{w,\mathrm{old}}+\Delta\mathbf v_w

이는 실제 force history를 만들지 않고 충격 직후의 state를 바로 생성한다. Policy가 외란을 받는 순간에는 이미
속도가 변해 있으므로 post-disturbance recovery state를 효율적으로 많이 학습하는 장점이 있다.

Mode 2: finite-duration force pulse
-----------------------------------------------

나머지 확률 ``0.5`` 에서는 duration을 표본화한다.

.. math::

   T\sim U(0.05,0.50)\quad [\mathrm{s}]

실제 구현은 policy period ``0.01 s`` 의 정수배인 5~50 step에서 균일 표본화한다. 목표 impulse가 velocity
mode와 같도록 환경별 전체 robot mass를 사용해 force를 계산한다.

.. math::

   \boxed{\mathbf F_w=\frac{m_{\mathrm{robot}}}{T}\Delta\mathbf v_w}

.. math::

   \mathbf J=\int_0^T\mathbf F_w\,dt
   =\mathbf F_wT
   =m_{\mathrm{robot}}\Delta\mathbf v_w

``m_robot`` 은 nominal 65 kg 상수가 아니라 startup mass DR 이후 각 environment의 실제 rigid-body mass 합이다.
따라서 같은 ``Delta v`` 와 ``T`` 에서 무거운 environment에는 더 큰 force가 들어가며, nominal momentum change는
environment mass에 맞게 유지된다. Force는 ``Base_Link`` 에 world-frame force로 적용하고 별도 torque는 0이다.

65 kg nominal mass 예시
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - ``Delta v_b`` 예
     - ``|Delta v|``
     - ``T=0.05 s``
     - ``T=0.50 s``
   * - pure rear ``[-0.5,0]``
     - ``0.500 m/s``
     - ``650 N``
     - ``65 N``
   * - pure front ``[+1.0,0]``
     - ``1.000 m/s``
     - ``1300 N``
     - ``130 N``
   * - rear diagonal ``[-0.5,+-0.5]``
     - ``0.707 m/s``
     - 약 ``919 N``
     - 약 ``92 N``
   * - front diagonal ``[+1.0,+-0.5]``
     - ``1.118 m/s``
     - 약 ``1453 N``
     - 약 ``145 N``

이는 applied base force다. 실제 root ``Delta v`` 는 발 접촉력, actuator PD, policy action, base 회전과 지면 반력의
영향을 동시에 받으므로 식의 목표값과 정확히 같지 않아도 정상이다. 바로 이 차이가 force mode에서 학습하려는
closed-loop recovery 과정이다.

시간 상태와 실행 순서
-----------------------------------------------

각 environment는 서로 독립적인 다음 push timer를 갖는다.

.. math::

   t_{\mathrm{next}}\sim U(10,15)\quad [\mathrm{s}]

Stateful event는 EventManager에서 policy period마다 호출되지만, 실제 외란은 이 timer가 0일 때만 시작한다.

.. code-block:: text

   episode reset
     -> active force를 0 N으로 해제
     -> next push time을 10~15 s에서 표본화

   each policy step (10 ms)
     -> active force duration 감소
     -> duration 종료 시 force를 정확히 0 N으로 해제
     -> push timer 감소
     -> timer가 0이면 Delta v_b 표본화
          -> Bernoulli(0.5)
             velocity: root velocity에 즉시 더함
             force: duration 표본화 후 F=m*Delta v/T 적용
          -> next push time 다시 10~15 s 표본화

Force maximum duration은 0.50 s이고 push minimum interval은 10 s이므로 정상 설정에서는 pulse가 겹치지 않는다.
그래도 구현은 새 disturbance 시작 전에 이전 active force를 명시적으로 지운다. Episode가 조기 종료되는 경우에도
``reset()`` 이 해당 environment의 force를 지우므로 다음 episode로 외력이 누출되지 않는다.

Freeze, command와의 관계
-----------------------------------------------

Command resampling, periodic freeze, episode timeout, mixed push timer는 서로 독립적이다. 현재 command role과
freeze 구성은 변경하지 않는다. 따라서 같은 외란은 다음 상태에서 모두 발생할 수 있다.

* 이동 command 추종 중
* exact-zero periodic freeze 중
* always-standing role
* freeze 진입 또는 해제 부근

Estimator가 Actor에 제공하는 estimated base linear velocity 덕분에 ``command=[0,0,0]`` 이어도 실제로 움직이는
recovery state와 안정된 standing state를 관측상 구별할 수 있다. 이번 branch는 stable/recovery reward gate를
새로 만들지 않고, 검증된 ``stand_still_joint_deviation_l1=-0.05`` 를 유지한다.

Standing weight ablation은 다른 모든 설정을 고정한 fresh 25k 학습으로 비교했다. Mixed-push baseline
``2026-08-24_17-10-31_concurrent_estimator_mixedpush_baseyaw_fresh25k`` 는 ``-0.05`` 를 사용한다. 후속
``2026-08-25_11-41-29_concurrent_estimator_mixedpush_stand001_fresh25k`` 는 ``-0.01`` 만 변경했지만 제자리
stepping이 나타났고, 최종 1,000 iteration 평균에서 action-rate와 actuator-torque 비용, illegal contact,
peak normal force가 모두 증가했다. 따라서 ``-0.01`` 은 제외하고 ``-0.05`` 를 mixed-push 기준값으로 복원했다.

Train과 Play/Teleop
-----------------------------------------------

.. list-table::
   :header-rows: 1

   * - 환경
     - 자동 mixed push
     - 수동 UI push
   * - Train
     - 활성, environment별 10~15 s
     - GUI를 띄운 학습이어도 사용하지 않음
   * - Play
     - 비활성
     - 활성
   * - Teleop
     - 비활성
     - 활성

Play/Teleop의 ``RoK4 Push Test`` 는 기존대로 base-yaw-frame 순간 velocity push만 제공한다. 따라서 checkpoint
평가에서 velocity recovery를 재현할 수 있지만, 학습 중 force pulse를 UI에서 그대로 재현하는 기능은 이번
변경 범위에 포함하지 않는다.

구현 파일과 검증
-----------------------------------------------

아래 경로는 모두 ``${ROK4_LAB_ROOT}`` 기준 상대경로다.

.. list-table::
   :header-rows: 1

   * - 파일
     - 역할
   * - ``.../velocity/mdp/events.py``
     - ``RoK4MixedPush``, frame 변환, mode 선택, force 계산과 해제
   * - ``.../config/rok4/domain_randomization_cfg.py``
     - 외란 범위, 확률, duration과 EventTerm 등록
   * - ``tests/test_events.py``
     - impulse 등가성, exclusive mode 선택, pulse 종료 검증

테스트는 다음 불변조건을 확인한다.

.. math::

   \mathbf F T=m\Delta\mathbf v

또한 한 event에서 velocity와 force가 동시에 적용되지 않는지, duration 종료와 reset에서 force가 0이 되는지를
확인한다. 실제 학습 전에는 GUI가 있는 소규모 environment에서 force 방향과 pulse 종료를 확인하고, 이후
headless 4096-env 학습으로 넘어가는 것이 좋다.

첫 실험에서 볼 항목
-----------------------------------------------

* 기존 estimator RMSE와 tracking reward가 유지되는가
* standing push에서 발을 내딛고 다시 양발 지지로 돌아오는가
* moving push에서 command tracking을 회복하는가
* ``+X`` 외란에서 오래 버티다 늦게 넘어지는 대신 조기에 전방 step을 선택하는가
* ``-X`` 외란이 과도해져 불필요한 후방 stepping을 만들지 않는가
* force duration이 긴 경우 몸을 계속 기울여 버티기만 하지 않는가
* 전방 평균 외란 때문에 normal gait와 standing COM이 편향되지 않는가

이 실험에서 reward, command population, estimator, gain을 동시에 바꾸지 않는다. 결과가 나쁘더라도 먼저
velocity mode와 force mode의 방향별 recovery를 분리해서 확인한 뒤, 외란 범위나 mode probability 중 한 변수만
조정한다.
