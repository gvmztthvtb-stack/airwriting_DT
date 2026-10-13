"""
ray_caster.py — Complementary Filter + 3D Ray-Cast 투영
========================================================

목적: 레이저 포인터처럼 손 방향 → 가상 벽면 교차점을 실시간 계산.

구성:
  1. ComplementaryRayCaster: 쿼터니언 기반 Complementary Filter
     - Pitch/Roll: 가속도계 중력 참조 (드리프트 프리)
     - Yaw: 순수 자이로 적분 (빠른 응답성)
     
  2. RayProjection: 쿼터니언 → 3D Ray → 2D 투영점 변환

왜 Complementary Filter인가?
  - 기존 MadgwickFilter(beta=0)는 가속도 보정 OFF → 100% 자이로 적분
  - 자이로만 사용하면 drift가 1~4°/분 누적 (ICM20948 스펙상 불가피)
  - Complementary Filter: 자이로의 빠른 응답성 + 가속도의 장기 안정성 융합
  - alpha=0.98 → 자이로 98% + 가속도 2% (미세하지만 drift 완전 상쇄)
"""

import numpy as np
import math
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


class ComplementaryRayCaster:
    """
    Complementary Filter 기반 방위 추정기.
    
    Pitch/Roll은 가속도계 중력 벡터로 보정 (노이즈/드리프트 제거).
    Yaw는 순수 자이로 적분 (가속도계로는 yaw 보정 불가).
    
    쿼터니언(w,x,y,z) 내부 표현 사용, Euler gimbal lock 회피.
    """
    
    def __init__(self, alpha: float = 0.98, sample_rate: float = 85.0):
        """
        Args:
            alpha: 자이로 신뢰 비율 (0~1). 0.98 = 자이로 98%, accel 2%.
                   높을수록 빠른 응답, 낮을수록 안정적.
            sample_rate: 센서 샘플링 주파수 (Hz)
        """
        self.alpha = alpha
        self.dt = 1.0 / sample_rate
        # 쿼터니언 [w, x, y, z] (scalar-first)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # 가속도 저역 통과 필터 (노이즈 제거)
        self._accel_lp = np.array([0.0, 0.0, 9.81], dtype=np.float64)
        self._accel_lp_alpha = 0.1  # LP 계수 (낮을수록 smooth)
        
    def update(self, accel: np.ndarray, gyro: np.ndarray, 
               dt: Optional[float] = None) -> np.ndarray:
        """
        센서 데이터로 방위 갱신.
        
        Args:
            accel: 가속도 [ax, ay, az] m/s²
            gyro:  각속도 [gx, gy, gz] rad/s (바이어스 보정 후)
            dt:    시간 간격 (None이면 기본값 사용)
            
        Returns:
            쿼터니언 [w, x, y, z]
        """
        step_dt = dt if dt is not None else self.dt
        
        # ── Step 1: 자이로 적분 (빠른 응답) ──
        q_gyro = self._integrate_gyro(gyro, step_dt)
        
        # ── Step 2: 가속도 중력 기반 pitch/roll (드리프트 프리) ──
        # 가속도 LP 필터 (진동/떨림 제거)
        self._accel_lp = (self._accel_lp_alpha * accel + 
                          (1.0 - self._accel_lp_alpha) * self._accel_lp)
        
        a_norm = np.linalg.norm(self._accel_lp)
        
        # 외부 가속이 심할 때(1G ±25%)는 중력 보정 스킵
        if a_norm < 7.36 or a_norm > 12.26:
            # 자이로만 사용 (안전)
            self.q = q_gyro
            self._normalize()
            return self.q
            
        # ── Step 3: Complementary 융합 ──
        # 가속도에서 roll, pitch 추출
        a_unit = self._accel_lp / a_norm
        pitch_accel = np.arctan2(-a_unit[0], np.sqrt(a_unit[1]**2 + a_unit[2]**2))
        roll_accel = np.arctan2(a_unit[1], a_unit[2])
        
        # 자이로 적분 쿼터니언에서 roll, pitch, yaw 추출
        yaw_gyro, pitch_gyro, roll_gyro = self._q_to_euler(q_gyro)
        
        # Complementary: pitch/roll은 accel과 혼합, yaw는 gyro 100%
        pitch_fused = self.alpha * pitch_gyro + (1.0 - self.alpha) * pitch_accel
        roll_fused = self.alpha * roll_gyro + (1.0 - self.alpha) * roll_accel
        yaw_fused = yaw_gyro  # yaw는 자이로만 (가속도에 yaw 정보 없음)
        
        # 융합된 Euler → 쿼터니언으로 복원
        self.q = self._euler_to_q(yaw_fused, pitch_fused, roll_fused)
        self._normalize()
        
        return self.q
    
    def _integrate_gyro(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        """자이로 적분: q_new = q * dq(gyro * dt)"""
        gx, gy, gz = gyro
        q0, q1, q2, q3 = self.q  # w, x, y, z
        
        # 쿼터니언 미분 (angular velocity → quaternion rate)
        qDot0 = 0.5 * (-q1*gx - q2*gy - q3*gz)
        qDot1 = 0.5 * ( q0*gx + q2*gz - q3*gy)
        qDot2 = 0.5 * ( q0*gy - q1*gz + q3*gx)
        qDot3 = 0.5 * ( q0*gz + q1*gy - q2*gx)
        
        q_new = np.array([
            q0 + qDot0 * dt,
            q1 + qDot1 * dt,
            q2 + qDot2 * dt,
            q3 + qDot3 * dt,
        ], dtype=np.float64)
        
        # 정규화
        norm = np.linalg.norm(q_new)
        if not np.isfinite(norm) or norm < 1e-9:
            return self.q.copy()
        return q_new / norm
    
    def _q_to_euler(self, q: np.ndarray) -> Tuple[float, float, float]:
        """쿼터니언 [w,x,y,z] → ZYX Euler [yaw, pitch, roll] (rad)"""
        w, x, y, z = q
        
        # yaw (Z축)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # pitch (Y축) — gimbal lock 방지 클램핑
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # roll (X축)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        return yaw, pitch, roll
    
    def _euler_to_q(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """ZYX Euler [yaw, pitch, roll] → 쿼터니언 [w, x, y, z]"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z], dtype=np.float64)
    
    def _normalize(self):
        """쿼터니언 정규화 (단위 쿼터니언 보장)"""
        norm = np.linalg.norm(self.q)
        if not np.isfinite(norm) or norm < 1e-9:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.q /= norm
    
    def reset(self, q_wxyz: Optional[np.ndarray] = None):
        """필터 상태 리셋"""
        if q_wxyz is not None:
            self.q = np.array(q_wxyz, dtype=np.float64)
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._accel_lp = np.array([0.0, 0.0, 9.81], dtype=np.float64)
    
    def get_yaw(self) -> float:
        """현재 yaw 각도 (rad) 반환"""
        yaw, _, _ = self._q_to_euler(self.q)
        return yaw


class RayProjection:
    """
    쿼터니언 → 2D 투영점 변환 (레이저 포인터 시뮬레이션).

    [원본 GitHub 동일 구현]
      - forward_local = [sin(15°), -cos(15°), 0]  ← Z=0 (XY 평면 안의 단위 벡터)
      - Roll(롤) 성분은 atan2 추출에서 자연스레 무시되어 Gimbal-Lock 없음
      - 25° pitch_tilt 같은 추가 오프셋을 두면 baseline phys_z ≠ 0 이 되어
        축이 섞이며 대각선 드리프트가 발생하므로 절대 추가하지 않음.

    반환값은 raw rad (스케일링은 호출부에서 일원화).
    """

    def __init__(self,
                 projection_distance: float = 2.5,
                 fov_limit_deg: float = 60.0,
                 deadzone_deg: float = 0.3):
        """
        Args:
            projection_distance: 가상 벽면까지 거리 (m).
            fov_limit_deg: 최대 시야각 (도).
            deadzone_deg: 데드존 (도). (현재 project()에서는 미적용)
        """
        self.distance = projection_distance
        self.fov_limit = np.radians(fov_limit_deg)
        self.deadzone = np.radians(deadzone_deg)

        # 투영 히트포인트 클램핑 (물리적 한계)
        self.max_hit = np.tan(self.fov_limit) * self.distance

        # 이전 히트 (데드존용)
        self._prev_hit = np.array([0.0, 0.0])

        # [원본 GitHub 동일] S3 검지 센서 좌측 15° 장착 보정 — Z 성분은 0 유지
        tilt_rad = np.radians(15.0)
        self._forward_local = np.array(
            [np.sin(tilt_rad), -np.cos(tilt_rad), 0.0]
        )

    def project(self, q_wxyz: np.ndarray) -> Tuple[float, float]:
        """
        쿼터니언 → 2D 투영 (unity_bridge_final_v1.py와 100% 수학적 정합).
        Scipy 대신 수동 쿼터니언 벡터 회전을 적용하여 축 비틀림 방지.
        """
        qw, qx, qy, qz = q_wxyz
        vx, vy, vz = self._forward_local

        # q * v * q_conj 수동 연산
        ix = qw * vx + qy * vz - qz * vy
        iy = qw * vy + qz * vx - qx * vz
        iz = qw * vz + qx * vy - qy * vx
        iw = -qx * vx - qy * vy - qz * vz

        rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
        rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
        forward = (rx, ry, rz)

        # Gimbal-Lock-Free atan2 기반 Yaw/Pitch 추출 (Roll 무시)
        phys_x = math.atan2(forward[0], -forward[1])                                  # Yaw (좌우) → X
        phys_z = math.atan2(forward[2], math.hypot(forward[0], forward[1]))           # Pitch (상하) → Y

        return (float(phys_x), float(phys_z))

    def reset(self):
        """투영 상태 리셋"""
        self._prev_hit = np.array([0.0, 0.0])


# =====================================================================
# Self-Test
# =====================================================================

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("  Ray Caster - Complementary Filter + 3D Ray Projection")
    print("=" * 60)
    
    # 1. ComplementaryRayCaster 테스트
    caster = ComplementaryRayCaster(alpha=0.98, sample_rate=85.0)
    
    # 정지 상태: 가속도 = [0, 0, 9.81], 자이로 = [0, 0, 0]
    for i in range(100):
        q = caster.update(
            np.array([0.0, 0.0, 9.81]),
            np.array([0.0, 0.0, 0.0]),
            dt=1/85.0
        )
    yaw, pitch, roll = caster._q_to_euler(q)
    print(f"\n  [Static Test] After 100 frames:")
    print(f"    Yaw: {np.degrees(yaw):.4f}°")
    print(f"    Pitch: {np.degrees(pitch):.4f}°")
    print(f"    Roll: {np.degrees(roll):.4f}°")
    assert abs(np.degrees(yaw)) < 0.1, "Yaw drift too high!"
    assert abs(np.degrees(pitch)) < 0.1, "Pitch drift too high!"
    print("    ✅ Static drift test PASSED")
    
    # 자이로 회전 테스트: Z축 1rad/s로 1초 회전
    caster.reset()
    for i in range(85):
        q = caster.update(
            np.array([0.0, 0.0, 9.81]),
            np.array([0.0, 0.0, 1.0]),  # 1 rad/s yaw
            dt=1/85.0
        )
    yaw, _, _ = caster._q_to_euler(q)
    print(f"\n  [Rotation Test] 1 rad/s for 1s:")
    print(f"    Expected yaw: {np.degrees(1.0):.1f}°")
    print(f"    Actual yaw: {np.degrees(yaw):.1f}°")
    error = abs(np.degrees(yaw) - np.degrees(1.0))
    print(f"    Error: {error:.2f}°")
    assert error < 3.0, f"Rotation error too high: {error:.2f}°"
    print("    ✅ Rotation test PASSED")
    
    # 2. RayProjection 테스트 — 원본 forward_local = [sin(15°), -cos(15°), 0]
    # project()는 raw rad 값을 반환 (스케일링은 main.py).
    yaw_tilt = np.radians(15.0)
    proj = RayProjection(projection_distance=2.5, fov_limit_deg=60.0, deadzone_deg=0.0)

    # 단위 쿼터니언 → (yaw_tilt, 0) (Z=0 이므로 phys_z baseline = 0)
    hx, hy = proj.project(np.array([1.0, 0.0, 0.0, 0.0]))
    print(f"\n  [Projection Test] Identity → phys=({np.degrees(hx):.2f}°, {np.degrees(hy):.2f}°)")
    assert abs(hx - yaw_tilt) < 0.01, f"Identity phys_x baseline mismatch: {np.degrees(hx)}"
    assert abs(hy) < 0.01, f"Identity phys_z should be 0: {np.degrees(hy)}"
    print("    ✅ Identity baseline test PASSED")

    # +Z 축 45° 회전 (yaw): phys_x 가 +45° 증가, phys_z 는 그대로 0
    yaw_45 = np.radians(45)
    cy = np.cos(yaw_45 / 2)
    sy = np.sin(yaw_45 / 2)
    q_45 = np.array([cy, 0.0, 0.0, sy])
    hx, hy = proj.project(q_45)
    print(f"  [Projection Test] +Z 45° → phys=({np.degrees(hx):.2f}°, {np.degrees(hy):.2f}°)")
    assert abs(hx - (yaw_tilt + yaw_45)) < 0.02, "Z-rot phys_x mismatch"
    assert abs(hy) < 0.02, "Z-rot should not change phys_z"
    print("    ✅ +Z rotation → phys_x test PASSED")

    # +X 축 30° 회전 (pitch): phys_z 가 변화, phys_x 는 거의 그대로
    proj.reset()
    pitch_30 = np.radians(30)
    cr = np.cos(pitch_30 / 2)
    sr = np.sin(pitch_30 / 2)
    q_pitch = np.array([cr, sr, 0.0, 0.0])
    # forward_local = [sin15, -cos15, 0]; +X rot 30° 후 직접 계산
    fl = np.array([np.sin(yaw_tilt), -np.cos(yaw_tilt), 0.0])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch_30), -np.sin(pitch_30)],
        [0.0, np.sin(pitch_30),  np.cos(pitch_30)],
    ])
    fwd = Rx @ fl
    expected_phys_x = np.arctan2(fwd[0], -fwd[1])
    expected_phys_z = np.arctan2(fwd[2], np.sqrt(fwd[0] ** 2 + fwd[1] ** 2))
    hx, hy = proj.project(q_pitch)
    print(f"  [Projection Test] +X 30° → phys=({np.degrees(hx):.2f}°, {np.degrees(hy):.2f}°)")
    assert abs(hx - expected_phys_x) < 0.02, f"+X rot phys_x mismatch"
    assert abs(hy - expected_phys_z) < 0.02, f"+X rot phys_z mismatch"
    print("    ✅ +X rotation → phys_z test PASSED")
    
    print("\n" + "=" * 60)
    print("  All tests PASSED!")
    print("=" * 60)

