# RIDA_version
Reliable Intelligent Diagnostic Assistant

버전 6기준
- 데이터 모듈 [실시간 데이터 생성, 데이터 전처리]
- 모델 모듈 [모델 로드, 구현 및 결과 처리]
- 인터페이스 모듈 [결과 디스플레이]

* 인터페이스 모듈
- 훈련 유무 진단 기능 [LSTM-AE]
- 비정상 시나리오 진단 기능 [LightGBM]
- 진단 결과 검증 기능 [LSTM-AE]
- 증상 만족 평가 기능 [Rule-based system]
- 진단 근거 도출 기능 [XAI-SHAP]

* 신규 추가 기능
- 로그 시스템 (실시간 구동을 보완하기 위해 이전 기록들을 기록하는 기능을 수행)

* 인터페이스 그림

![Interface](https://user-images.githubusercontent.com/56631737/107496497-bba00f80-6bd4-11eb-8d88-49fb7d29a0a2.png)


* 로그 시스템 그림

![Log_system](https://user-images.githubusercontent.com/56631737/107496641-e8ecbd80-6bd4-11eb-9e47-b62e545256e7.png)

위의 훈련 유무 진단 기능, 비정상 시나리오 진단 기능, 진단 결과 검증 기능의 결과들을 기록하여 저장하는 기능임.
* 그래프
- 훈련 유무 진단 기능 -> 0: 훈련된 상태, 1: 훈련되지 않은 상태
- 비정상 시나리오 진단 기능 -> 0: 정상 상태, 1~15: 비정상 상태
- 진단 결과 검증 기능 -> 0: 진단 성공, 1: 진단 실패

* 왼쪽 패널
- 변화가 있을 경우에만 프린트됨. (Log Update)
- 각 기능의 전체 기록은 아래의 버튼을 클릭하면 제공됨.
