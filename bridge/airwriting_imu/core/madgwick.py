import numpy as np
import math

class MadgwickFilter:
    """
    순수 쿼터니언 기반 Madgwick AHRS 필터 (IMU 모드 - 가속도 + 자이로)
    Euler 각도를 우회하여 짐벌락 및 축 간 크로스 커플링(대각선 드리프트)을 방지합니다.
    """
    
    def __init__(self, beta: float = 0.05, sample_rate: float = 85.0):
        self.beta = beta
        self.dt = 1.0 / sample_rate
        # 쿼터니언 [w, x, y, z] (scalar-first)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
    def reset(self, q_wxyz: np.ndarray = None):
        """방위 리셋"""
        if q_wxyz is not None:
            self.q = np.array(q_wxyz, dtype=np.float64)
            self.q /= np.linalg.norm(self.q)
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            
    def update_imu(self, accel: np.ndarray, gyro: np.ndarray, dt: float = None) -> np.ndarray:
        """
        자이로와 가속도를 이용해 방위 업데이트.
        
        Args:
            accel: 가속도 [ax, ay, az] (m/s^2 또는 임의의 단위, 내부에서 정규화됨)
            gyro: 각속도 [gx, gy, gz] (rad/s)
            dt: 이번 스텝의 시간 (None일 경우 초기화된 sample_rate 사용)
            
        Returns:
            갱신된 쿼터니언 [w, x, y, z]
        """
        step_dt = dt if dt is not None else self.dt
        
        q0, q1, q2, q3 = self.q  # w, x, y, z
        gx, gy, gz = gyro
        ax, ay, az = accel
        
        # 가속도 정규화
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm == 0.0:
            # 가속도가 0이면 자이로만 적분
            return self._integrate_gyro(gx, gy, gz, step_dt)
            
        ax /= norm
        ay /= norm
        az /= norm
        
        # 보조 변수
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        
        # Gradient descent 목적 함수 f
        f1 = _2q1 * q3 - _2q0 * q2 - ax
        f2 = _2q0 * q1 + _2q2 * q3 - ay
        f3 = 1.0 - _2q1 * q1 - _2q2 * q2 - az

        # 자코비안 J와 f의 곱으로 step 계산 (Gradient)
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
        s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay
        
        # Step 정규화
        norm_s = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if norm_s > 0.0:
            s0 /= norm_s
            s1 /= norm_s
            s2 /= norm_s
            s3 /= norm_s
            
        # 자이로 기반 쿼터니언 변화율
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * ( q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * ( q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * ( q0 * gz + q1 * gy - q2 * gx)
        
        # Gradient descent 결과를 반영한 최종 변화율
        qDot1 -= self.beta * s0
        qDot2 -= self.beta * s1
        qDot3 -= self.beta * s2
        qDot4 -= self.beta * s3
        
        # 적분
        q0 += qDot1 * step_dt
        q1 += qDot2 * step_dt
        q2 += qDot3 * step_dt
        q3 += qDot4 * step_dt
        
        # 정규화
        norm_q = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        self.q = np.array([q0 / norm_q, q1 / norm_q, q2 / norm_q, q3 / norm_q], dtype=np.float64)
        
        return self.q
        
    def _integrate_gyro(self, gx, gy, gz, dt) -> np.ndarray:
        q0, q1, q2, q3 = self.q
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * ( q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * ( q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * ( q0 * gz + q1 * gy - q2 * gx)
        
        self.q = np.array([
            q0 + qDot1 * dt,
            q1 + qDot2 * dt,
            q2 + qDot3 * dt,
            q3 + qDot4 * dt
        ], dtype=np.float64)
        self.q /= np.linalg.norm(self.q)
        return self.q
