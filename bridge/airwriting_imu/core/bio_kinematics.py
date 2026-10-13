"""
Bio Kinematics -- ICOR Modeling + Differential Kinematics
==========================================================

정밀 HCI를 위한 바이오 키네마틱스 기반 동작 분리.

핵심 구성:
  1. ICORModel: 순간 회전 중심(ICOR) 트래커 -- MCP/PIP/DIP 관절
  2. DifferentialKinematics: 3-IMU 차동 운동학 (드리프트 프리)
  3. MotionSeparator: 손목/손가락 움직임 분리기

물리적 참조:
  - PIP 관절: 가장 큰 동적 RoM + 최고 속도
  - 구름-미끄러짐(Rolling-sliding) 결합으로 ICOR 지속 이동
  - 3-IMU: S1(forearm), S2(hand), S3(finger)
"""

import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from typing import Tuple, Optional


# =====================================================================
# ICOR Model: Instantaneous Center of Rotation Tracker
# =====================================================================

class ICORModel:
    """
    순간 회전 중심(Instantaneous Center of Rotation) 모델링.
    
    실제 손가락 관절은 고정축 회전이 아닌 구름-미끄러짐 결합.
    ICOR이 지속적으로 이동하며, 이를 무시하면 운동학적 정렬 불량 발생.
    
    IRM(Instant Radius Method): 의료 영상 기반 ICOR 궤적 연속 모델링.
    """
    
    # 해부학적 참조값 (성인 남성 평균, mm)
    JOINT_PARAMS = {
        'MCP': {  # Metacarpophalangeal (손가락 뿌리)
            'radius_range': (8.0, 12.0),   # mm
            'rom_flex': (-10, 90),          # degrees
            'rom_abd': (-15, 15),
            'max_velocity': 8.0,            # rad/s
        },
        'PIP': {  # Proximal Interphalangeal (중간 마디)
            'radius_range': (5.0, 8.0),
            'rom_flex': (0, 110),           # 가장 큰 RoM
            'max_velocity': 12.0,           # 가장 높은 최고 속도
        },
        'DIP': {  # Distal Interphalangeal (끝 마디)
            'radius_range': (3.0, 5.0),
            'rom_flex': (0, 80),
            'max_velocity': 6.0,
        },
    }
    
    def __init__(self, joint_type: str = 'PIP'):
        params = self.JOINT_PARAMS[joint_type]
        self.joint_type = joint_type
        self.r_min, self.r_max = params['radius_range']
        self.rom = params.get('rom_flex', (0, 90))
        self.max_vel = params.get('max_velocity', 10.0)
        
        # ICOR 상태
        self.current_angle = 0.0       # rad
        self.current_radius = (self.r_min + self.r_max) / 2.0
        self.icor_position = np.zeros(2)  # 2D ICOR 위치 (관절 로컬)
        self.trajectory = deque(maxlen=500)
    
    def update(self, angle_rad: float) -> np.ndarray:
        """
        관절 각도 업데이트 -> ICOR 위치 계산.
        
        IRM: 관절 각도에 따라 회전 반경이 비선형적으로 변화.
        r(theta) = r_min + (r_max - r_min) * sigmoid(k * theta)
        """
        self.current_angle = angle_rad
        
        # 비선형 반경 모델 (sigmoid 기반)
        norm_angle = (angle_rad - np.radians(self.rom[0])) / (
            np.radians(self.rom[1] - self.rom[0]) + 1e-6)
        norm_angle = np.clip(norm_angle, 0, 1)
        
        k = 5.0  # sigmoid 경사도
        sigmoid_val = 1.0 / (1.0 + np.exp(-k * (norm_angle - 0.5)))
        self.current_radius = self.r_min + (self.r_max - self.r_min) * sigmoid_val
        
        # ICOR 위치 (관절 좌표계, mm -> m)
        r_m = self.current_radius / 1000.0
        self.icor_position = np.array([
            -r_m * np.sin(angle_rad),
            r_m * np.cos(angle_rad)
        ])
        
        self.trajectory.append(self.icor_position.copy())
        
        return self.icor_position
    
    def get_corrected_transform(self, angle_rad: float) -> np.ndarray:
        """ICOR 보정된 2D 변환 행렬."""
        icor = self.update(angle_rad)
        
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        # 회전 중심이 ICOR인 변환
        T = np.eye(3)
        T[0, 0] = c;  T[0, 1] = -s
        T[1, 0] = s;  T[1, 1] = c
        T[0, 2] = icor[0] - (c * icor[0] - s * icor[1])
        T[1, 2] = icor[1] - (s * icor[0] + c * icor[1])
        return T


# =====================================================================
# Differential Kinematics: 3-IMU Joint Angle Computation
# =====================================================================

class DifferentialKinematics:
    """
    3-IMU 차동 운동학 알고리즘.
    
    두 개 이상의 센서 간 상대적인 가속도/각속도 차이로 관절 각도 직접 계산.
    적분 드리프트 취약점을 중력/자기장 벡터로 보정.
    
    물리 센서 배치 (실측, 코드 S1=전완을 원점, 단위: mm):
      코드 S1 (wrist/forearm):  (  0,  0,   0)  ← 원점
      코드 S2 (hand dorsum):    (-29, 23, 143)
      코드 S3 (index finger):   (-44, 20, 229)  ← 좌측 15도 기울어짐 (검지 해부 구조)
    
    주의: 화이트보드 라벨(S1=검지, S3=전완)과 코드 라벨(S1=전완, S3=검지)이 반대!
    좌표계: X=오른쪽, Y=전방(손가락 방향), Z=위(팔 방향)
    """
    
    # ─── 실측 센서 좌표 (mm, 코드 컨벤션: S1=전완=원점) ───
    SENSOR_POSITIONS_MM = {
        's1': np.array([  0.0,  0.0,   0.0]),   # 전완부 (코드 S1 = 화이트보드 S3)
        's2': np.array([-29.0, 23.0, 143.0]),   # 손등
        's3': np.array([-44.0, 20.0, 229.0]),   # 검지 (코드 S3 = 화이트보드 S1)
    }
    
    # 센서 간 벡터 (mm, 코드 컨벤션)
    SENSOR_VECTORS_MM = {
        's1_to_s2': np.array([-29.0, 23.0, 143.0]),  # 전완→손등 (손목 관절)
        's2_to_s3': np.array([-15.0, -3.0,  86.0]),   # 손등→검지 (MCP/PIP 관절)
    }
    
    # 실측 뼈 길이 (m)
    BONE_LENGTHS = {
        'forearm_to_hand': 0.1477,    # S1→S2: sqrt(29²+23²+143²) = 147.7mm
        'hand_to_finger': 0.0874,     # S2→S3: sqrt(15²+3²+86²)  =  87.4mm
        'finger_to_tip': 0.020,       # S3 센서~검지 끝 (약 20mm)
    }
    
    # S3(검지) 센서 기울기 보정 (좌측 15도, 검지 해부학적 구조)
    S3_TILT_DEG = -15.0  # 음수 = 좌측
    
    def __init__(self):
        # ICOR 모델 (각 관절)
        self.icor_mcp = ICORModel('MCP')
        self.icor_pip = ICORModel('PIP')
        self.icor_dip = ICORModel('DIP')
        
        # S3 기울기 보정 회전 행렬 (Z축 기준 -15도)
        tilt_rad = np.radians(self.S3_TILT_DEG)
        self.s3_tilt_correction = Rotation.from_euler('Z', tilt_rad)
        
        # 관절 각도 상태
        self.joint_angles = {
            'wrist_flex': 0.0,
            'wrist_abd': 0.0,
            'mcp_flex': 0.0,
            'pip_flex': 0.0,
            'dip_flex': 0.0,
        }
        
        # 차동 신호 필터 (저역 통과)
        self.alpha_lp = 0.3  # 저역 통과 계수
        self.diff_accel_s1s2 = np.zeros(3)  # S1-S2 전용
        self.diff_gyro_s1s2 = np.zeros(3)
        self.diff_accel_s2s3 = np.zeros(3)  # S2-S3 전용
        self.diff_gyro_s2s3 = np.zeros(3)
        
        # 절대 방위 (드리프트 프리 보정용)
        self.gravity_ref = np.array([0, 0, 9.81])
    
    def compute_differential(
        self,
        a_proximal: np.ndarray, g_proximal: np.ndarray,
        a_distal: np.ndarray, g_distal: np.ndarray,
        q_proximal: Rotation = None, q_distal: Rotation = None,
        dt: float = 0.01,
        pair: str = 's1s2'
    ) -> dict:
        """
        차동 센싱: 두 인접 센서의 상대 운동 계산.
        
        pair: 's1s2' 또는 's2s3' — 필터 상태 선택
        Returns: dict with 'diff_accel', 'diff_gyro', 'joint_angle', 'joint_velocity'
        """
        # 1. 고해상도 차동 센싱
        diff_accel = a_distal - a_proximal
        diff_gyro = g_distal - g_proximal
        
        # 저역 통과 필터 (관절 쌍별 독립 상태)
        if pair == 's1s2':
            self.diff_accel_s1s2 = self.alpha_lp * diff_accel + (1 - self.alpha_lp) * self.diff_accel_s1s2
            self.diff_gyro_s1s2 = self.alpha_lp * diff_gyro + (1 - self.alpha_lp) * self.diff_gyro_s1s2
            filtered_a, filtered_g = self.diff_accel_s1s2, self.diff_gyro_s1s2
        else:
            self.diff_accel_s2s3 = self.alpha_lp * diff_accel + (1 - self.alpha_lp) * self.diff_accel_s2s3
            self.diff_gyro_s2s3 = self.alpha_lp * diff_gyro + (1 - self.alpha_lp) * self.diff_gyro_s2s3
            filtered_a, filtered_g = self.diff_accel_s2s3, self.diff_gyro_s2s3
        
        # 2. 관절 각속도 (차동 자이로)
        joint_velocity = np.linalg.norm(filtered_g)
        
        # 3. 관절 각도 추정 (중력 기반)
        joint_angle = 0.0
        if q_proximal is not None and q_distal is not None:
            # 쿼터니언 상대 회전 (q_rel = q_proximal.inv() * q_distal)
            # scipy Rotation 객체에서 직접 쿼터니언 [x, y, z, w] 추출
            q_rel = (q_proximal.inv() * q_distal).as_quat()
            # XYZ 오일러 각 중 Y축(flex) 성분을 직접 계산 (asin(2*(w*y - z*x)))
            # as_euler() 대비 약 10배 이상 빠름
            qx, qy, qz, qw = q_rel
            siny = 2.0 * (qw * qy - qz * qx)
            joint_angle = np.arcsin(np.clip(siny, -1.0, 1.0))
        else:
            # 중력 벡터 기반 근사 (폴백)
            a_p_norm = a_proximal / (np.linalg.norm(a_proximal) + 1e-8)
            a_d_norm = a_distal / (np.linalg.norm(a_distal) + 1e-8)
            cos_angle = np.clip(np.dot(a_p_norm, a_d_norm), -1, 1)
            joint_angle = np.arccos(cos_angle)
        
        return {
            'diff_accel': filtered_a.copy(),
            'diff_gyro': filtered_g.copy(),
            'joint_angle': float(joint_angle),
            'joint_velocity': float(joint_velocity),
        }
    
    def update_full_chain(
        self,
        s1_accel: np.ndarray, s1_gyro: np.ndarray,
        s2_accel: np.ndarray, s2_gyro: np.ndarray,
        s3_accel: np.ndarray, s3_gyro: np.ndarray,
        q_s1: Rotation = None, q_s2: Rotation = None, q_s3: Rotation = None,
        dt: float = 0.01
    ) -> dict:
        """
        전체 3-IMU 체인 업데이트.
        
        S1-S2: 손목 관절 (wrist)
        S2-S3: MCP + PIP 합산 (손가락 굽힘)
        """
        # 손목 관절 (S1 -> S2)
        wrist = self.compute_differential(
            s1_accel, s1_gyro, s2_accel, s2_gyro, q_s1, q_s2, dt, pair='s1s2')
        self.joint_angles['wrist_flex'] = wrist['joint_angle']
        
        # 손가락 관절 (S2 -> S3)
        finger = self.compute_differential(
            s2_accel, s2_gyro, s3_accel, s3_gyro, q_s2, q_s3, dt, pair='s2s3')
        self.joint_angles['mcp_flex'] = finger['joint_angle'] * 0.6  # MCP 기여
        self.joint_angles['pip_flex'] = finger['joint_angle'] * 0.4  # PIP 기여
        
        # ICOR 보정
        self.icor_mcp.update(self.joint_angles['mcp_flex'])
        self.icor_pip.update(self.joint_angles['pip_flex'])
        
        return {
            'wrist': wrist,
            'finger': finger,
            'joint_angles': dict(self.joint_angles),
            'icor_mcp': self.icor_mcp.icor_position.tolist(),
            'icor_pip': self.icor_pip.icor_position.tolist(),
        }


# =====================================================================
# Motion Separator: Wrist/Finger motion decomposition
# =====================================================================

class MotionSeparator:
    """
    손목의 큰 움직임과 손가락의 미세한 움직임을 분리.
    
    대역 통과 필터 + 차동 신호 융합.
    에어라이팅 시 손목 drift 제거, 손가락 미세 궤적만 추출.
    """
    def __init__(self, sample_rate: float = 85.0):
        self.sample_rate = sample_rate
        
        # 대역 분리 (손목: 0-3Hz, 손가락: 3-20Hz)
        self.wrist_cutoff = 3.0   # Hz
        self.finger_low = 3.0
        self.finger_high = 20.0
        
        # 1차 IIR 필터 상태
        alpha_w = 2 * np.pi * self.wrist_cutoff / sample_rate
        self.alpha_wrist = alpha_w / (alpha_w + 1)
        
        self.wrist_filtered = np.zeros(3)
        self.finger_filtered = np.zeros(3)
        self.prev_raw = np.zeros(3)
    
    def separate(self, s1_accel: np.ndarray, s3_accel: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        S1(wrist)과 S3(finger) 가속도에서 동작 성분 분리.
        
        Returns: (wrist_motion, finger_motion)
          wrist_motion: 저주파 큰 움직임
          finger_motion: 고주파 미세 궤적
        """
        # 차동 신호
        diff = s3_accel - s1_accel
        
        # 저역 통과 (손목 성분)
        self.wrist_filtered = (
            self.alpha_wrist * s1_accel +
            (1 - self.alpha_wrist) * self.wrist_filtered
        )
        
        # 고역 통과 (손가락 성분) = 전체 - 저역
        self.finger_filtered = diff - (
            self.alpha_wrist * diff +
            (1 - self.alpha_wrist) * self.finger_filtered
        )
        
        self.prev_raw = diff.copy()
        
        return self.wrist_filtered.copy(), self.finger_filtered.copy()
    
    def get_writing_intent(self, finger_motion: np.ndarray) -> float:
        """손가락 미세 동작으로부터 필기 의도(Writing Intent) 점수 추출."""
        energy = np.sum(finger_motion ** 2)
        return float(np.clip(energy / 5.0, 0, 1))
    
    def reset(self):
        self.wrist_filtered = np.zeros(3)
        self.finger_filtered = np.zeros(3)
        self.prev_raw = np.zeros(3)


# =====================================================================
# Self-Test
# =====================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    print("=" * 60)
    print("  Bio Kinematics - ICOR + Differential Kinematics")
    print("=" * 60)
    
    # ICOR Test
    icor = ICORModel('PIP')
    for angle_deg in range(0, 110, 10):
        pos = icor.update(np.radians(angle_deg))
        print(f"  PIP {angle_deg:3d}deg -> ICOR: [{pos[0]*1000:.2f}, {pos[1]*1000:.2f}] mm, "
              f"radius: {icor.current_radius:.1f} mm")
    
    # Differential Kinematics Test
    dk = DifferentialKinematics()
    result = dk.update_full_chain(
        s1_accel=np.array([0, 0, 9.81]),
        s1_gyro=np.zeros(3),
        s2_accel=np.array([0, 1.0, 9.76]),
        s2_gyro=np.array([0, 0.1, 0]),
        s3_accel=np.array([0.5, 2.0, 9.5]),
        s3_gyro=np.array([0, 0.3, 0]),
    )
    print(f"\nJoint Angles: {result['joint_angles']}")
    print(f"ICOR MCP: {result['icor_mcp']}")
    print(f"ICOR PIP: {result['icor_pip']}")
    
    # Motion Separator Test
    sep = MotionSeparator()
    for i in range(100):
        s1 = np.array([0, 0, 9.81]) + np.random.randn(3) * 0.1
        s3 = np.array([np.sin(i*0.2), np.cos(i*0.2), 9.81]) + np.random.randn(3) * 0.05
        wrist, finger = sep.separate(s1, s3)
    
    intent = sep.get_writing_intent(finger)
    print(f"\nWriting Intent: {intent:.3f}")
    print("\nDone!")
