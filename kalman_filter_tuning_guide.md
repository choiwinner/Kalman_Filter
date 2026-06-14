# 칼만 필터 파라미터 (Q, R) 최적화 및 튜닝 가이드

칼만 필터의 성능은 시스템의 내재적 변동을 뜻하는 **시스템 잡음 분산 ($Q$)**과 센서의 특성을 뜻하는 **측정 잡음 분산 ($R$)**에 의해 결정됩니다. 본 가이드는 실무에서 고정된 최적의 $Q$와 $R$ 값을 구하고 설정하는 구체적인 방법을 다룹니다.

---

## 1. 측정 잡음 분산 ($R$) 설정 및 측정 방법

$R$은 센서의 노이즈 특성을 대변하며, **실제 센서 데이터를 획득하여 직접 수학적으로 도출**할 수 있습니다.

### 1) 실측을 위한 실험 방법
1. **시스템 고정**: 전류 측정 시스템을 정상 가동하되, 상태 변화(전류의 증감)가 전혀 없도록 완전히 일정한 상태로 고정합니다. (예: 대기 상태 혹은 고정된 저항 부하 연결 상태)
2. **원시 데이터(Raw data) 수집**: 센서 샘플링 속도에 따라 **최소 500개에서 1000개 이상의 원시 측정값**을 로그 파일 등으로 기록합니다.
3. **분산 연산**: 획득한 데이터의 **분산(Variance, $\sigma^2$)**을 계산합니다.

### 2) 파이썬을 이용한 R 산출 예시 코드
```python
import numpy as np

# 1. 정지 상태에서 센서로부터 수집한 1000개의 전류 측정 데이터 예시
static_measured_data = np.array([49.8, 52.1, 48.5, 50.2, 51.5, 49.0, 50.8, 52.3, 47.9, 50.1]) # ... (실제 데이터 1000개 입력)

# 2. 표본 데이터의 분산(Variance) 계산
R_measured = np.var(static_measured_data)

print(f"계산된 측정 잡음 분산 (R): {R_measured:.4f}")
# 만약 센서 데이터의 흔들림 표준편차가 2.0mA 수준이었다면 R은 약 4.0 내외로 계산됩니다.
```

---

## 2. 시스템 잡음 분산 ($Q$) 설정 및 튜닝 방법

$Q$는 직접 측정이 불가능한 불확실성 값이므로, 사전에 고정한 $R$ 값을 바탕으로 **시뮬레이션 분석(Grid Search)과 실무 테스트를 통해 튜닝**해야 합니다.

### 1) 그리드 서치(Grid Search)를 통한 최적화 튜닝
실제 전류의 참값(Ground Truth)을 모사하거나 알고 있는 실험 환경에서 여러 $Q$ 값을 시도해 보고, **참값과의 평균 제곱 오차(MSE, Mean Squared Error)를 최소화**하는 $Q$를 탐색합니다.

### 2) 파이썬을 이용한 최적 Q 자동 탐색 코드
```python
import numpy as np

def run_kalman(measurements, Q, R, initial_est, initial_err):
    """1차원 칼만 필터를 수행하는 함수"""
    x_est = initial_est
    P = initial_err
    estimates = []
    
    for z in measurements:
        # 1. 예측 (Prediction)
        x_pred = x_est
        P_pred = P + Q
        
        # 2. 업데이트 (Update)
        K = P_pred / (P_pred + R)
        x_est = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        estimates.append(x_est)
        
    return np.array(estimates)

# 시뮬레이션 환경 구축
true_current = 50.0 # 실제 참값
np.random.seed(42)
# R_fixed = 4.0(표준편차 2.0의 제곱)인 잡음 데이터 1000개 생성
measurements = true_current + np.random.normal(0, 2.0, 1000) 
R_fixed = 4.0 # 앞선 단계에서 실측한 분산값으로 고정

# 최적의 Q를 찾기 위한 후보군 설정 (로그 스케일: 10^-6부터 10^0까지)
Q_candidates = np.logspace(-6, 0, 7)
best_Q = None
min_mse = float('inf')

for Q in Q_candidates:
    filtered = run_kalman(measurements, Q, R_fixed, initial_est=50.0, initial_err=1.0)
    
    # 참값(50.0)과의 평균 제곱 오차(MSE) 계산
    mse = np.mean((filtered - true_current) ** 2)
    print(f"Q 후보군: {Q:.6f} | 평균 제곱 오차(MSE): {mse:.4f}")
    
    if mse < min_mse:
        min_mse = mse
        best_Q = Q

print(f"\n[결과] 최적의 시스템 잡음 분산 Q: {best_Q:.6f} (최소 MSE: {min_mse:.4f})")
```

---

## 3. 실무 하드웨어 적용 시 튜닝 검토사항

실무 현장에서는 참값(Ground Truth)을 모르는 상태에서 튜닝을 진행해야 하는 경우가 많습니다. 이때는 다음과 같은 정성적/정량적 지표를 활용합니다.

### 1) 지연 시간(Time Delay) 모니터링
* 시스템 전류를 **의도적으로 급격하게 변동(Step Input)** 시킵니다. (예: 10mA에서 50mA로 즉각 전환)
* 필터링된 출력이 변동된 실제 목표값에 도달하는 데 걸리는 **반응 지연 시간(상승 시간)**을 측정합니다.
* 노이즈 차단력과 응답 속도 사이의 최적 타협점을 찾기 위해, **허용 가능한 응답 속도 범위 내에서 가장 작은 $Q$**를 채택하는 것이 정석입니다.

### 2) 주파수 영역 분석 (FFT 분석)
* 필터를 거치기 전의 원시 데이터와 거친 후의 데이터를 주파수 변환(FFT)하여, 시스템에 해가 되는 전원 노이즈 대역(예: 60Hz 대역 등)이 목표 비율 이상 감쇄되었는지 스펙트럼 형태로 확인하며 $Q$를 튜닝합니다.
