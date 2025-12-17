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
│   ├── train/           #훈련용 데이터                       
│   │   ├── rainy/       # 훈련용 비오는 이미지            
│   │   │   ├── 001.png                                                      
│   │   │   └── ...                                  
│   │   └── clean/  # 훈련용 비가 오지 않는 환경 이미지                                    
│   │       ├── 001.png                       
│   │       └── ...                              
│   └── validation/   # 검증용 데이터                           
│       ├── rainy/                                
│       │   ├── 101.png     # 검증용 비오는 이미지                        
│       │   └── ...                               
│       └── clean/     # 검증용 비가 오지 않는 환경 이미지                     
│           ├── 101.png                                 
│           └── ...                                    
├── run_test.py                                   
├── run_test_image.py                                   
├── train.py                                            
└── .venv/

이미지 쌍 이름 반드시 동일
훈련용 이미지 8 : 검증용 이미지 2 정도의 비율 (조절 가능) 과적합 방지
## lane detection
1. 외부적 요소를 줄이기위한 roi영역 지정 및 BEV변환                         
2. 와이퍼 영역 제거를 위한 모폴로지 마스크                                   
3. HSV + HSL + sobel필터를 통해 노란색 및 흰색 차선에 해당하는 edge추출                 
4. Sliding window + ransac 알고리즘을 통한 차선 추출                     
5. 차선 인식 실패시 기존 차선을 이용한 추적                                   
### test
1.새로운 영역을 지정            
2.지정한 영역내에 존재하는 차선의 좌표를 필터링
3.기존 코드에서 검출한 좌표들과 비교하여 우측 좌측 차선을 각각 차선 인식률 비교
4.백분율로 좌측과 우측의 차선의 인식률을 각각 표현
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
물웅덩이 문제 - 기존 kalman filter이용, clache, 추가 필터등 추가적 알고리즘을 통한 해결책을 시행해 보았지만 눈에 띄는 개선상항이 없었다. RRBC를 통한 딥러닝 기법에 추가적 조치를 취하는 것이 가장 효과적일 것으로 보인다.

와이퍼 - 모폴로지 기법을 통해 와이퍼 부분을 제외하고 차선을 인식할 수 있도록 하였지만 test코드와 함께 돌렸을때 이부분의 인식률이 순간적으로 낮아지거나 끊기는 것은 남아있었다.
