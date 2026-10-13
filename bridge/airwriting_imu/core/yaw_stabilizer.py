"""
yaw_stabilizer.py — 3중 Yaw 드리프트 보정 시스템
=================================================

Madgwick 6축 필터(update_imu)는 중력만 사용하므로 Roll/Pitch만 보정.
Yaw(수평 회전)는 보정 수단이 없어 자이로 적분 오차가 누적.

본 모듈은 3가지 독립 보정 메커니즘을 융합하여 Yaw 드리프트를 근본 제거:

  1. AdaptiveMagFusion: 자기장 이상 감지 + 선택적 Yaw 보정
  2. GyroBiasEstimator: 정지 상태(ZARU) 감지 → 자이로 바이어스 실시간 보정
  3. DualSensorYawAnchor: S1(전완) 대비 S3(검지) 상대 Yaw 앵커링

사용법:
  stabilizer = YawStabilizer(sample_rate=85.0)
  # 매 프레임:
  corrected_gyro = stabilizer.process(
      accel, gyro, mag,        # S3 센서 데이터
      s1_accel, s1_gyro,       # S1 센서 데이터
      is_writing               # 필기 상태
  )
  # corrected_gyro를 Madgwick에 넣으면 Yaw 드리프트 제거
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Adaptive Magnetometer Fusion
# =====================================================================

class AdaptiveMagFusion:
    """
    자기장 이상 감지 + 선택적 Yaw 보정.
    
    실내 환경에서 자기장이 왜곡되면 Swimming이 발생.
    해결: 자기장의 크기와 방향이 '정상 범위' 내일 때만 Yaw 보정.
    
    판단 기준:
      1. 자기장 크기(norm): 캘리브레이션 시 기준값 대비 ±30% 이내
      2. 자기장 방향(dip angle): 기준 대비 ±15도 이내
      3. 변화율: 프레임 간 갑작스런 변화 거부
    """
    
    def __init__(self, sample_rate: float = 85.0):
        self.sample_rate = sample_rate
        
        # 기준값 (캘리브레이션에서 설정)
        self.ref_norm: Optional[float] = None
        self.ref_dip: Optional[float] = None  # 복각 (수직 성분 비율)
        self.ref_heading: Optional[float] = None  # 초기 Yaw 방향
        
        # 허용 범위
        self.norm_tolerance = 0.30     # ±30%
        self.dip_tolerance_deg = 15.0  # ±15도
        self.rate_limit = 50.0         # µT/s 최대 변화율
        
        # 이력
        self.prev_mag = None
        self.trust_score = 0.0   # 0~1: 현재 자기장 신뢰도
        self.trust_ema = 0.0     # EMA 스무딩된 신뢰도
        self.ema_alpha = 0.05    # 매우 느린 EMA (이상 시 즉시 거부, 복구는 서서히)
        
        # 통계
        self.total_samples = 0
        self.accepted_samples = 0
        
    def calibrate(self, mag_samples: np.ndarray):
        """
        캘리브레이션 시 수집된 자기장 데이터로 기준값 설정.
        
        mag_samples: shape (N, 3)
        """
        norms = np.linalg.norm(mag_samples, axis=1)
        self.ref_norm = float(np.median(norms))
        
        # 복각: arctan(Z / sqrt(X²+Y²))
        horizontal = np.sqrt(mag_samples[:, 0]**2 + mag_samples[:, 1]**2)
        dips = np.degrees(np.arctan2(mag_samples[:, 2], horizontal + 1e-8))
        self.ref_dip = float(np.median(dips))
        
        logger.info(f"[MagFusion] Calibrated: norm={self.ref_norm:.1f}µT, dip={self.ref_dip:.1f}°")
        
    def set_reference_heading(self, heading_rad: float):
        """캘리브레이션 완료 시 초기 Yaw 방향 저장."""
        self.ref_heading = heading_rad
        
    def evaluate(self, mag: np.ndarray, dt: float) -> float:
        """
        자기장 데이터의 신뢰도를 0~1로 평가.
        
        Returns: trust_score (0=완전 거부, 1=완전 신뢰)
        """
        if self.ref_norm is None or mag is None:
            return 0.0
            
        self.total_samples += 1
        
        mag_norm = np.linalg.norm(mag)
        if mag_norm < 1e-6:
            self.trust_score = 0.0
            return 0.0
            
        # --- 테스트 1: 크기 검증 ---
        norm_ratio = mag_norm / self.ref_norm
        norm_ok = abs(norm_ratio - 1.0) < self.norm_tolerance
        norm_score = max(0, 1.0 - abs(norm_ratio - 1.0) / self.norm_tolerance)
        
        # --- 테스트 2: 복각 검증 ---
        horizontal = np.sqrt(mag[0]**2 + mag[1]**2)
        current_dip = np.degrees(np.arctan2(mag[2], horizontal + 1e-8))
        dip_error = abs(current_dip - self.ref_dip)
        dip_ok = dip_error < self.dip_tolerance_deg
        dip_score = max(0, 1.0 - dip_error / self.dip_tolerance_deg)
        
        # --- 테스트 3: 변화율 검증 ---
        rate_score = 1.0
        if self.prev_mag is not None and dt > 0:
            rate = np.linalg.norm(mag - self.prev_mag) / dt
            rate_score = max(0, 1.0 - rate / self.rate_limit)
        self.prev_mag = mag.copy()
        
        # 종합 점수 (최소값 기반 — 하나라도 나쁘면 거부)
        self.trust_score = min(norm_score, dip_score, rate_score)
        
        # EMA: 거부는 즉각, 복구는 서서히
        alpha = 0.5 if self.trust_score < self.trust_ema else self.ema_alpha
        self.trust_ema = alpha * self.trust_score + (1 - alpha) * self.trust_ema
        
        if self.trust_ema > 0.5:
            self.accepted_samples += 1
            
        return self.trust_ema
    
    def compute_yaw_correction(self, mag: np.ndarray, current_q_wxyz: np.ndarray) -> float:
        """
        자기장에서 Yaw 오차를 계산.
        
        Returns: yaw_correction (rad) — 양수면 반시계 보정 필요
        """
        if self.ref_heading is None:
            return 0.0
            
        # 현재 쿼터니언으로 자기장을 월드 프레임으로 회전
        qw, qx, qy, qz = current_q_wxyz
        
        # 센서 자기장 → 월드 좌표
        mx, my, mz = mag
        # 쿼터니언 회전 (q * v * q^-1)
        # 간소화: 수평면 성분만 사용
        # 월드 좌표에서의 수평 자기장 방향 = heading
        hx = (mx * (1 - 2*qy*qy - 2*qz*qz) + 
              my * (2*qx*qy - 2*qw*qz) + 
              mz * (2*qx*qz + 2*qw*qy))
        hy = (mx * (2*qx*qy + 2*qw*qz) + 
              my * (1 - 2*qx*qx - 2*qz*qz) + 
              mz * (2*qy*qz - 2*qw*qx))
        
        current_heading = np.arctan2(hy, hx)
        
        # 기준 대비 Yaw 오차
        yaw_error = current_heading - self.ref_heading
        
        # ±π 정규화
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
            
        return float(yaw_error)
        
    def get_stats(self) -> dict:
        acceptance = self.accepted_samples / max(1, self.total_samples) * 100
        return {
            'trust': round(self.trust_ema, 3),
            'acceptance_pct': round(acceptance, 1),
            'ref_norm': self.ref_norm,
        }


# =====================================================================
# 2. Gyro Bias Estimator (ZARU)
# =====================================================================

class GyroBiasEstimator:
    """
    Zero Angular Rate Update (ZARU).
    
    정지 상태를 감지하고 자이로 바이어스를 실시간으로 추정/보정.
    특히 Yaw 축 바이어스가 드리프트의 주범.
    
    정지 판단:
      - 가속도 norm ≈ 9.81 m/s² (±0.3)
      - 자이로 norm < threshold (rad/s)
      - 연속 N프레임 이상 유지
    """
    
    def __init__(self, sample_rate: float = 85.0):
        self.sample_rate = sample_rate
        
        # 정지 감지 임계값
        self.accel_g_range = (9.51, 10.11)  # m/s² (9.81 ± 0.3)
        self.gyro_threshold = 0.05           # rad/s (≈3°/s)
        self.min_static_frames = 10          # 85Hz에서 ~120ms
        
        # 상태
        self.static_count = 0
        self.is_static = False
        
        # 바이어스 추정 (EMA)
        self.bias = np.zeros(3)           # 현재 추정된 바이어스
        self.bias_alpha = 0.05            # 캘리브레이션이 대부분 잡은 뒤 잔여분만 추적
        self.bias_locked = False          # 필기 중엔 잠금
        
        # 윈도우 기반 분산 추정
        self.gyro_window = []
        self.window_size = 30  # ~350ms
        
    def update(self, accel: np.ndarray, gyro: np.ndarray, is_writing: bool) -> np.ndarray:
        """
        자이로 바이어스를 보정한 gyro 반환.
        
        Returns: bias-corrected gyro (3,)
        """
        a_norm = np.linalg.norm(accel)
        g_norm = np.linalg.norm(gyro - self.bias)
        
        # 정지 판단
        accel_static = self.accel_g_range[0] <= a_norm <= self.accel_g_range[1]
        gyro_static = g_norm < self.gyro_threshold
        
        if accel_static and gyro_static and not is_writing:
            self.static_count += 1
            if self.static_count >= self.min_static_frames:
                self.is_static = True
                self._update_bias(gyro)
        else:
            self.static_count = 0
            self.is_static = False
            
        # 바이어스 보정된 자이로 반환
        return gyro - self.bias
    
    def _update_bias(self, gyro: np.ndarray):
        """정지 시 자이로 평균으로 바이어스 업데이트."""
        self.gyro_window.append(gyro.copy())
        if len(self.gyro_window) > self.window_size:
            self.gyro_window.pop(0)
            
        if len(self.gyro_window) >= self.window_size // 2:
            window_mean = np.mean(self.gyro_window, axis=0)
            window_std = np.std(self.gyro_window, axis=0)
            
            # 분산이 충분히 작을 때만 업데이트 (노이즈 아닌 진짜 정지)
            if np.all(window_std < 0.02):
                self.bias = self.bias_alpha * window_mean + (1 - self.bias_alpha) * self.bias
    
    def get_bias(self) -> np.ndarray:
        return self.bias.copy()
    
    def get_stats(self) -> dict:
        return {
            'is_static': self.is_static,
            'bias_deg_s': np.degrees(self.bias).tolist(),
            'static_count': self.static_count,
        }


# =====================================================================
# 3. Dual-Sensor Yaw Anchor
# =====================================================================

class DualSensorYawAnchor:
    """
    S1(전완) 대비 S3(검지) 상대 Yaw 앵커링.
    
    원리: 전완은 에어라이팅 중 비교적 안정적.
    S1의 Yaw 변화가 작은데 S3의 Yaw가 크게 변하면 → 드리프트로 판단.
    
    구현:
      1. S1/S3 각각의 Yaw를 독립 추적
      2. 상대 Yaw 차이(S3-S1)가 물리적 한계(±90°) 초과 시 보정
      3. 느린 드리프트: S1 Yaw 변화율과 S3 Yaw 변화율 비교
    """
    
    def __init__(self, sample_rate: float = 85.0):
        self.sample_rate = sample_rate
        
        # S1 독립 Madgwick (안정 참조용)
        self.s1_yaw = 0.0
        self.s3_yaw = 0.0
        self.relative_yaw_ref = 0.0  # 캘리브레이션 시 상대 Yaw
        
        # S1 자이로 적분 (Z축 = Yaw)
        self.s1_yaw_rate_ema = 0.0
        self.s3_yaw_rate_ema = 0.0
        self.rate_ema_alpha = 0.1
        
        # 보정 강도
        self.max_relative_yaw = np.radians(90)  # 물리적 한계
        self.correction_gain = 0.002  # 매우 부드러운 보정
        self.drift_gain = 0.001       # 느린 드리프트 보정
        
        # 초기화 플래그
        self.initialized = False
        
    def calibrate(self, s1_yaw: float, s3_yaw: float):
        """캘리브레이션 시 상대 Yaw 기준점 설정."""
        self.s1_yaw = s1_yaw
        self.s3_yaw = s3_yaw
        self.relative_yaw_ref = s3_yaw - s1_yaw
        self.initialized = True
        logger.info(f"[YawAnchor] Calibrated: s1={np.degrees(s1_yaw):.1f}°, "
                    f"s3={np.degrees(s3_yaw):.1f}°, relative={np.degrees(self.relative_yaw_ref):.1f}°")
        
    def update(self, s1_gyro: np.ndarray, s3_gyro: np.ndarray, dt: float) -> float:
        """
        S1 기준 S3의 Yaw 보정량 계산.
        
        Returns: yaw_correction (rad) — S3 자이로 Z축에 더해줄 값
        """
        if not self.initialized:
            return 0.0
            
        # Yaw 변화율 (Z축 자이로)
        s1_yaw_rate = s1_gyro[2]
        s3_yaw_rate = s3_gyro[2]
        
        # EMA 스무딩
        self.s1_yaw_rate_ema = (self.rate_ema_alpha * s1_yaw_rate + 
                                 (1 - self.rate_ema_alpha) * self.s1_yaw_rate_ema)
        self.s3_yaw_rate_ema = (self.rate_ema_alpha * s3_yaw_rate + 
                                 (1 - self.rate_ema_alpha) * self.s3_yaw_rate_ema)
        
        # Yaw 적분
        self.s1_yaw += s1_yaw_rate * dt
        self.s3_yaw += s3_yaw_rate * dt
        
        correction = 0.0
        
        # --- 보정 1: 물리적 한계 초과 ---
        relative_yaw = self.s3_yaw - self.s1_yaw
        deviation = relative_yaw - self.relative_yaw_ref
        
        if abs(deviation) > self.max_relative_yaw:
            # 급격한 보정 (물리적으로 불가능한 각도)
            correction = -deviation * self.correction_gain * 10
            
        # --- 보정 2: 느린 드리프트 감지 ---
        # S1이 거의 안 움직이는데 S3 Yaw가 서서히 변하면 → 드리프트
        s1_still = abs(self.s1_yaw_rate_ema) < 0.02  # ~1°/s
        s3_drifting = abs(self.s3_yaw_rate_ema) > 0.005  # ~0.3°/s
        
        if s1_still and s3_drifting:
            # S3의 드리프트 성분을 S1 방향으로 끌어당김
            correction += -self.s3_yaw_rate_ema * self.drift_gain
            
        return float(correction)
    
    def get_stats(self) -> dict:
        rel = self.s3_yaw - self.s1_yaw
        return {
            'relative_yaw_deg': round(np.degrees(rel), 1),
            'ref_deg': round(np.degrees(self.relative_yaw_ref), 1),
            'deviation_deg': round(np.degrees(rel - self.relative_yaw_ref), 1),
        }


# =====================================================================
# 4. Unified Stabilizer
# =====================================================================

class YawStabilizer:
    """
    3중 보정을 통합한 Yaw 안정화 시스템.
    
    사용법:
        stabilizer = YawStabilizer(sample_rate=85.0)
        
        # 캘리브레이션 후:
        stabilizer.calibrate(mag_samples, s1_yaw, s3_yaw, heading)
        
        # 매 프레임:
        corrected_gyro = stabilizer.process(
            s3_accel, s3_gyro, s3_mag,
            s1_gyro, is_writing, current_q, dt
        )
    """
    
    def __init__(self, sample_rate: float = 85.0):
        self.mag_fusion = AdaptiveMagFusion(sample_rate)
        self.bias_estimator = GyroBiasEstimator(sample_rate)
        self.yaw_anchor = DualSensorYawAnchor(sample_rate)
        
        self.sample_rate = sample_rate
        self._correction_log = []
        
    def calibrate(self, 
                  mag_samples: Optional[np.ndarray] = None,
                  s1_yaw: float = 0.0, 
                  s3_yaw: float = 0.0,
                  heading: float = 0.0):
        """캘리브레이션 결과로 모든 보정기 초기화."""
        if mag_samples is not None and len(mag_samples) > 5:
            self.mag_fusion.calibrate(mag_samples)
            self.mag_fusion.set_reference_heading(heading)
        self.yaw_anchor.calibrate(s1_yaw, s3_yaw)
        
    def process(self,
                s3_accel: np.ndarray,
                s3_gyro: np.ndarray,
                s3_mag: Optional[np.ndarray],
                s1_gyro: np.ndarray,
                is_writing: bool,
                current_q_wxyz: Optional[np.ndarray] = None,
                dt: float = 0.0117) -> np.ndarray:
        """
        3중 보정을 적용한 S3 자이로 반환.
        
        Returns: corrected_s3_gyro (3,) — Madgwick에 직접 넣을 값
        """
        # Step 1: ZARU — Z축(Yaw) 바이어스만 선택적 보정
        # Roll/Pitch는 Madgwick 중력 경사하강이 자동 보정하므로 건드리지 않습니다.
        # Z축(Yaw)만 빼줘서 "한쪽으로 서서히 휘는" 드리프트를 차단합니다.
        # (이전에 전 축 보정 → Roll/Pitch 크로스커플링 → Jumping이었으므로 Z축만 적용)
        _ = self.bias_estimator.update(s3_accel, s3_gyro, is_writing)
        z_bias = float(np.clip(self.bias_estimator.bias[2], -0.05, 0.05))
        
        # Step 2: (비활성) DualSensorYawAnchor — PCB 안정화 후 재평가

        # Step 3: 적응형 자기장 Yaw 보정 (유지)
        mag_yaw_correction = 0.0
        if s3_mag is not None and current_q_wxyz is not None:
            trust = self.mag_fusion.evaluate(s3_mag, dt)
            
            if trust > 0.5:
                yaw_error = self.mag_fusion.compute_yaw_correction(s3_mag, current_q_wxyz)
                mag_gain = 0.01 * trust  # 최대 0.01 rad/s 보정 (캘리브레이션이 대부분 잡으므로 강화 가능)
                mag_yaw_correction = -yaw_error * mag_gain
                # 한 프레임당 최대 보정량 제한 (갑작스런 튐 방지)
                mag_yaw_correction = float(np.clip(mag_yaw_correction, -0.003, 0.003))
        
        # 합성: Z축(Yaw)에만 ZARU 바이어스 보정 + 자기장 미세 보정
        corrected = s3_gyro.copy()
        corrected[2] = corrected[2] - z_bias + mag_yaw_correction
        
        return corrected
    
    def get_stats(self) -> dict:
        return {
            'bias': self.bias_estimator.get_stats(),
            'anchor': {'deviation_deg': 0.0, 'status': 'disabled'},
            'mag': self.mag_fusion.get_stats(),
        }


# =====================================================================
# Self-Test
# =====================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    print("=" * 60)
    print("  Yaw Stabilizer - 3-Layer Drift Correction")
    print("=" * 60)
    
    stab = YawStabilizer(sample_rate=85.0)
    
    # Fake calibration
    mag_samples = np.random.randn(50, 3) * 5 + np.array([20, 5, 40])
    stab.calibrate(mag_samples, s1_yaw=0.0, s3_yaw=0.0, heading=0.0)
    
    # Simulate 1000 frames with slow yaw drift
    drift_rate = 0.001  # rad/s (tiny but accumulates)
    total_drift = 0.0
    corrected_drift = 0.0
    
    for i in range(1000):
        accel = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.05
        gyro = np.array([0, 0, drift_rate]) + np.random.randn(3) * 0.005
        s1_gyro = np.random.randn(3) * 0.005
        mag = np.array([20, 5, 40]) + np.random.randn(3) * 0.5
        
        dt = 1/85.0
        total_drift += drift_rate * dt
        
        corrected = stab.process(accel, gyro, mag, s1_gyro, False, np.array([1,0,0,0]), dt)
        corrected_drift += corrected[2] * dt
    
    print(f"\n  Raw drift after 1000 frames: {np.degrees(total_drift):.2f}°")
    print(f"  Corrected drift:             {np.degrees(corrected_drift):.2f}°")
    print(f"  Reduction:                   {(1 - abs(corrected_drift/total_drift))*100:.1f}%")
    print(f"\n  Stats: {stab.get_stats()}")
    print("\nDone!")
