# Summary
현재 자율주행 시스템에서는 비가오는 환경에서 여러가지 방해요소들로 인해 차선인식률이 떨어지는 문제점이 있다. 이를 딥러닝기법과 알고리즘적 처리 기법을 합친 하이브리드 기법을 통해 개선하고자 하였다.

정적인 환경에서 성능이 그렇게 떨어지지 않고 속도가 빠르다는 장점을 살리기 위해 딥러닝 기법을 통해 영상을 정적인 환경으로 만들어주고 이 영상을 알고리즘적 처리를 통해 차선을 인식시키면 자원소모량도 기존 딥러닝 기법보다 적으면서 적절한 성능도 얻을 수 있지않을까라는 예측을 기반으로 시작하였다.

RRBC기법을
# code instruction
RRBC (CNN+LSTM) -> lane detection(HSV filter + sliding window + track + ransac)
## RRBC
### train
### test
### file instruction
RRBC/
├── data/
│   ├── train/
│   │   ├── rainy/ 
│   │   │   ├── 001.png
│   │   │   └── ...
│   │   └── clean/
│   │       ├── 001.png
│   │       └── ...
│   └── validation/
│       ├── rainy/ 
│       │   ├── 101.png
│       │   └── ...
│       └── clean/
│           ├── 101.png
│           └── ...
├── run_test.py
├── run_test_image.py
├── train.py
└── .venv/

## lane detection
### test
# Demo
<img width="1436" height="813" alt="image" src="https://github.com/user-attachments/assets/229018e5-051c-43db-b458-df22b2a17dc0" />

# 결론 및 남은 과제
### track
<img width="1147" height="530" alt="스크린샷 2025-12-17 130314" src="https://github.com/user-attachments/assets/2ceb361d-5d5c-49fd-9c2c-ecab945ebe31" />
### kalman filter
<img width="1148" height="528" alt="스크린샷 2025-12-17 130539" src="https://github.com/user-attachments/assets/1cd475b3-421a-404f-b7be-f5c2a72f9b34" />
전체적으로 학습량을 증가 시킬 수록 차선인식률이 증가되는 모습을 볼 수 있다.

충분한 학습을 진행할 시 기존 lane detection만을 진행하는 것보다 뛰어난 성능을 얻을 수 있을 것이다.
## 남은 문제점
