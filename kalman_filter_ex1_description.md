# 1차원 단순 칼만 필터 예제 설명서 (`kalman_filter_ex1.ipynb`)

이 문서는 `kalman_filter_ex1.ipynb` 노트북 파일에서 구현된 1차원 단순 칼만 필터(Simple Kalman Filter)의 동작 원리, 코드 구조 및 파라미터 튜닝 가이드를 설명합니다.

---

## 1. 개요
본 예제는 시스템 상태(State)가 하나인 **1차원 상황**에서 노이즈가 섞인 센서 계측 데이터를 칼만 필터를 통해 어떻게 효과적으로 필터링하는지 보여줍니다.
* **가상의 시나리오**: 실제 전류가 50mA로 일정하게 흐르는 DC-DC 컨버터 환경에서, 표준편차 $2.0\text{mA}$의 잡음(Noise)이 추가된 센서 측정값으로부터 실제 전류 상태를 실시간 추정합니다.
* **목적**: 불확실한 시스템 모델 예측과 노이즈가 포함된 센서 측정값을 적절히 융합하여 최적의 상태를 추정하는 것입니다.

---

## 2. 칼만 필터 알고리즘 단계 (1차원 모델)
칼만 필터는 **예측(Prediction)** 단계와 **업데이트(Update)** 단계가 매 시간 단계마다 반복되는 순환 구조를 가집니다.

### (1) 예측 단계 (Prediction)
이전 단계의 추정값을 바탕으로 현재 단계의 상태를 예측합니다.
* **상태 예측**:
  $$x_{\text{pred}, k} = x_{\text{est}, k-1}$$
  *(1차원 상수를 가정하므로 이전 추정 상태를 그대로 사용합니다.)*
* **오차 분산 예측**:
  $$P_{\text{pred}, k} = P_{k-1} + Q$$
  *(이전 추정 오차 분산 $P$에 시스템 자체의 변동성/불확실성인 프로세스 노이즈 분산 $Q$를 더해 불확실성이 증가함을 나타냅니다.)*

### (2) 업데이트 단계 (Update)
새로운 센서 측정값을 반영하여 상태를 보정하고 최적의 값을 추정합니다.
* **칼만 이득(Kalman Gain) 계산**:
  $$K_k = \frac{P_{\text{pred}, k}}{P_{\text{pred}, k} + R}$$
  * 예측값의 불확실성($P_{\text{pred}}$)과 센서 측정 오차 분산($R$) 간의 비율을 뜻합니다.
  * $R$이 매우 크면(센서가 부정확하면) $K$는 0에 가까워져 센서 측정을 무시하고 예측값에 의존합니다.
  * $P_{\text{pred}}$가 매우 크면(예측 모델이 부정확하면) $K$는 1에 가까워져 센서 측정값을 강하게 반영합니다.
* **추정 상태 업데이트**:
  $$x_{\text{est}, k} = x_{\text{pred}, k} + K_k \cdot (z_k - x_{\text{pred}, k})$$
  *(예측한 값에 '측정값($z_k$)과 예측값의 차이(잔차, Residual)'에 칼만 이득을 곱한 값을 더해 상태를 보정합니다.)*
* **추정 오차 분산 업데이트**:
  $$P_k = (1 - K_k) \cdot P_{\text{pred}, k}$$
  *(필터링을 거치면서 상태의 불확실성이 줄어들게 되므로, 오차 분산 $P$를 감소시킵니다.)*

---

## 3. 주요 매개변수(Parameter) 및 튜닝 방법

칼만 필터의 성능은 다음 파라미터들을 어떻게 설정하느냐에 따라 크게 달라집니다.

| 파라미터 | 기호 | 의미 | 튜닝 가이드 |
| :--- | :--- | :--- | :--- |
| **Process Noise** | $Q$ | 시스템 모델의 불확실성 분산 | 시스템이 안정적이면 아주 작은 값($1e^{-5} \sim 1e^{-3}$)을 설정하여 부드럽게 필터링합니다. 시스템의 변화 속도가 빠르면 값을 키워($1e^{-2} \sim 0.5$) 센서 반응을 빠르게 추종하도록 합니다. |
| **Measurement Noise** | $R$ | 센서 측정값의 잡음 분산 | 센서 사양서의 오차 범위나 표준편차($\sigma$)의 제곱값($\sigma^2$)을 직접 입력합니다. 잡음이 심할수록 이 값을 크게 설정하여 필터가 센서 흔들림에 둔감해지도록 만듭니다. |
| **Initial Estimate** | $x_{\text{est}}$ | 초기 추정 상태값 | 센서 구동이 시작되는 시점의 예측 시작값입니다. 대개 최초 센서 측정값($z_0$)을 그대로 할당하여 시작합니다. |
| **Initial Error** | $P$ | 초기 추정 오차 분산 | 초기 추정치의 불확실성 정도입니다. 시작 지점 값을 신뢰한다면 작게($0.1 \sim 1.0$), 신뢰할 수 없다면 크게($100.0 \sim 1000.0$) 잡습니다. 단, 필터가 반복되면서 빠르게 최적의 값으로 수렴하므로 최종 성능에 큰 미치지는 않습니다. |

---

## 4. 코드 구현 구조

### (1) `SimpleKalmanFilter` 클래스
1차원 칼만 필터 연산을 캡슐화한 파이썬 클래스입니다.

```python
class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_estimate, initial_error):
        self.Q = process_noise               # 시스템 변동 오차 분산
        self.R = measurement_noise           # 센서 측정 오차 분산
        self.x_est = initial_estimate       # 상태 추정치 초기값
        self.P = initial_error              # 추정 오차 분산 초기값

    def update(self, measurement):
        # 1. 예측 단계 (Prediction)
        x_pred = self.x_est
        P_pred = self.P + self.Q

        # 2. 업데이트 단계 (Update)
        # 칼만 이득 계산
        K = P_pred / (P_pred + self.R)
        
        # 최적 추정 상태 보정 및 오차 분산 갱신
        self.x_est = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return self.x_est
```

### (2) 시뮬레이션 및 데이터 생성
실제 전류값 50.0mA에 백색 잡음(정규분포 오차)을 인위적으로 더하여 100개의 센서 데이터 샘플을 생성합니다.

```python
true_current = 50.0 
np.random.seed(42)
# 평균 0, 표준편차 2.0의 정규분포 노이즈를 더함 (분산 R = 2.0^2 = 4.0)
measurements = true_current + np.random.normal(0, 2.0, 100)

# 필터 인스턴스 초기화 (Q=0.001, R=4.0)
kf = SimpleKalmanFilter(process_noise=1e-3, measurement_noise=4.0, initial_estimate=50.0, initial_error=4.0)

# 실시간 업데이트 루프 실행
filtered_current = [kf.update(z) for z in measurements]
```

---

## 5. 시각화 및 결과 해석

* `matplotlib`와 `koreanize-matplotlib`를 사용하여 필터링 전후의 전류 추이를 차트로 시각화합니다.
* **필터 효과**: 오차가 큰 원본 센서 측정 신호(파란색 실선)가 칼만 필터를 통과하면서 실제 참값인 50mA(초록색 점선)에 매우 부드럽고 가깝게 추정값(빨간색 실선)으로 필터링되는 것을 관찰할 수 있습니다.
* **튜닝 영향**:
  * $Q$값을 낮추거나 $R$값을 늘리면 필터 그래프가 지연(Delay)이 생기며 매우 부드러워집니다.
  * $Q$값을 높이거나 $R$값을 줄이면 노이즈는 조금 덜 제거되지만 실제 값의 급격한 변화에 빠르게 적응(Fast Response)합니다.
