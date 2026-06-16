┌─────────────────────────────────────────────────────────────┐
│                     데이터 수집 과정                          │
└─────────────────────────────────────────────────────────────┘

  [하드웨어]                [ROS2]               [스크립트]

  RealSense D435  ──────►  /camera/color     ──┐
                           /image_raw           │
                                                ├──► collect_data.py
  UR5e           ──────►  /joint_states      ──┤       │
  (제어 스크립트            (6 joint angles)      │       │ s키: 녹화 시작
   로 동작 중)                                   │       │ e키: 종료 & 저장
                                                │       │
  Robotiq 2F-85  ──────►  /joint_states      ──┘       │
                           (finger_joint)               │
                                                        │ TF2
                           base → tool0  ◄──────────────┘
                           (EEF 포즈 계산)


  [저장]

  episode_0000.hdf5
  ├── observations/images/primary   (T, 256, 256, 3)  RGB 이미지
  ├── observations/proprio          (T, 7)             조인트6 + 그리퍼
  └── actions                       (T, 7)             delta EEF(6) + 그리퍼
  episode_0001.hdf5
  ...
  episode_0099.hdf5   ← 목표: 50~100개


  [다음 단계]

  collected_data/
       │
       ▼
  RLDS 변환 (rlds_dataset_builder)
       │
       ▼
  finetune.py  (OpenVLA-OFT 파인튜닝)
       │
       ▼
  deploy.py  (실제 로봇 배포)

  
