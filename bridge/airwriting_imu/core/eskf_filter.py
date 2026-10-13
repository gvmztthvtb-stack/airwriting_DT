import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation

class ESKF:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.g = np.array([0, 0, 9.81])
        
        # Nominal State
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = Rotation.from_quat([0, 0, 0, 1]) # scalar-last [x,y,z,w]
        self.a_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self.mag_ref_world = None
        
        # Error State Covariance (15x15)
        self.P = np.eye(15) * 0.01
        
        # Process Noise Covariances (Base values)
        self.Q_base = np.eye(12) 
        self.Q_base[0:3, 0:3] *= 0.05    # accel noise
        self.Q_base[3:6, 3:6] *= 0.01    # gyro noise
        self.Q_base[6:9, 6:9] *= 0.001   # accel bias random walk
        self.Q_base[9:12, 9:12] *= 0.0001# gyro bias random walk
        self.Q = np.copy(self.Q_base)
        
        # Measurement Covariance (ZUPT)
        self.R_zupt = np.eye(3) * 0.05
        self.R_zaru = np.eye(3) * 0.001
        
        # ZUPT Window
        self.window_size = 15
        self.a_win = deque(maxlen=15)
        self.g_win = deque(maxlen=15)

        # Pre-allocated matrices to avoid per-frame heap churn (3 ESKF × 85Hz)
        self._Fx = np.eye(15)
        self._I15 = np.eye(15)
        self._Fi = np.zeros((15, 12))
        self._Fi[6:9, 3:6] = np.eye(3)
        self._Fi[9:12, 6:9] = np.eye(3)
        self._Fi[12:15, 9:12] = np.eye(3)

        # P 행렬 대각선 상한 (장시간 무한 성장 방지)
        # 원본 [1,0.5,0.1,0.05,0.01]은 버벅임, [10,5,1,0.5,0.1]은 노이즈 과다
        # 타협: pos/vel은 3배 완화(반응성), att/bias는 원본 근처(노이즈 억제)
        self._p_max = np.array(
            [3.0]*3 + [1.5]*3 + [0.2]*3 + [0.1]*3 + [0.02]*3,
            dtype=np.float64
        )
        
    def reset(self, initial_q=None, initial_ba=None, initial_bg=None, initial_mag=None):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        if initial_q is not None:
            # calibration.py가 준 q_align은 World -> Sensor 회전입니다. 
            # ESKF 내부에서 self.q는 Sensor -> World 여야 하므로 역행렬(inv)을 취합니다.
            self.q = Rotation.from_quat(initial_q).inv()
        else:
            self.q = Rotation.from_quat([0, 0, 0, 1])
            
        if initial_ba is not None:
            self.a_b = np.asarray(initial_ba, dtype=np.float64)
        else:
            self.a_b = np.zeros(3)

        if initial_bg is not None:
            self.w_b = np.asarray(initial_bg, dtype=np.float64)
        else:
            self.w_b = np.zeros(3)
            
        # [Phase 5] 지자기 참조점 (World Frame) 저장
        if initial_mag is not None:
            self.mag_ref_world = self.q.as_matrix() @ initial_mag
        else:
            self.mag_ref_world = None
        self.P = np.eye(15) * 0.01
        self.a_win.clear()
        self.g_win.clear()

    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def predict(self, accel, gyro, dt):
        # 1. Update Nominal State
        accel_true = accel - self.a_b
        gyro_true = gyro - self.w_b
        # Orientation update
        angle = np.linalg.norm(gyro_true) * dt
        if angle > 1e-8:
            axis = gyro_true / np.linalg.norm(gyro_true)
            dq = Rotation.from_rotvec(axis * angle)
            try:
                self.q = self.q * dq
            except Exception:
                self.q = Rotation.identity()
            
            # 쿼터니언 수치적 복원 및 재정규화
            q_val = self.q.as_quat()
            if np.isnan(q_val).any() or np.isinf(q_val).any() or np.linalg.norm(q_val) < 1e-6:
                self.q = Rotation.identity()
            else:
                # 명시적 재정규화로 크기 0 방지
                q_val = q_val / np.linalg.norm(q_val)
                self.q = Rotation.from_quat(q_val)
            
        R = self.q.as_matrix()
        
        # Velocity & Position update
        accel_world = R @ accel_true - self.g
            
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt
        
        # 2. Update Error State Covariance (in-place writes on pre-allocated buffers)
        # 2. Update Error State Covariance (Sparse block-wise update to save CPU)
        # P = Fx @ P @ Fx.T + Q_eff
        # Fx is identity except for a few blocks. We can update P in-place or with minimal copies.
        
        # Extract blocks for speed
        P = self.P
        
        # Temporary updates for non-zero off-diagonal blocks of Fx
        d_v_q = -R @ self._skew(accel_true) * dt
        d_v_ba = -R * dt
        d_q_q = -self._skew(gyro_true) * dt  # F_qq = I + d_q_q
        d_q_bw = -np.eye(3) * dt
        
        # 1. Update Rows (P_new = Fx @ P)
        P_old = P.copy()
        # Row p: P[0:3, :] = P[0:3, :] + dt * P[3:6, :]
        P[0:3, :] += dt * P_old[3:6, :]
        # Row v: P[3:6, :] = P[3:6, :] + d_v_q @ P[6:9, :] + d_v_ba @ P[9:12, :]
        P[3:6, :] += d_v_q @ P_old[6:9, :] + d_v_ba @ P_old[9:12, :]
        # Row q: P[6:9, :] = P[6:9, :] + d_q_q @ P[6:9, :] + d_q_bw @ P[12:15, :]
        P[6:9, :] += d_q_q @ P_old[6:9, :] + d_q_bw @ P_old[12:15, :]
        
        # 2. Update Columns (P_final = P_new @ Fx.T)
        P[:, 0:3] += dt * P[:, 3:6]
        P[:, 3:6] += P[:, 6:9] @ d_v_q.T + P[:, 9:12] @ d_v_ba.T
        P[:, 6:9] += P[:, 6:9] @ d_q_q.T + P[:, 12:15] @ d_q_bw.T
        
        # 3. Add Process Noise (Fi @ Q @ Fi.T) * dt^2
        dt2 = dt * dt
        P[3:6, 3:6] += (R @ self.Q[0:3, 0:3] @ R.T) * dt2
        P[6:9, 6:9] += self.Q[3:6, 3:6] * dt2
        P[9:12, 9:12] += self.Q[6:9, 6:9] * dt2
        P[12:15, 12:15] += self.Q[9:12, 9:12] * dt2
        
        # [수치 안전장치] NaN 발생 시 0으로 복구하고 극단적 값 클리핑
        if np.isnan(P).any() or np.isinf(P).any():
            P[np.isnan(P)] = 0.0
            P[np.isinf(P)] = 0.0
        np.clip(P, -1e6, 1e6, out=P)
        self.a_win.append(accel)
        self.g_win.append(gyro)
            
        # 4. Adaptive Q & Damping & Clamping
        # ZUPT 상태 점검 (predict() 직전에 밖에서 세팅될 수도 있으므로 여기서 방어적 처리)
        is_stationary = self.detect_zupt()
        if is_stationary:
            self.Q = self.Q_base * 0.1
            self.v *= 0.90 # 정지 상태 강력한 제동
        else:
            self.Q = self.Q_base * 2.0
            self.v *= 0.995 # 초당 ~65% 보존 (85Hz 기준: 0.995^85 ≈ 0.65)
            
        # 속도 클램핑 (최대 2.0 m/s 제한, 인간 필기 속도 한계)
        v_norm = np.linalg.norm(self.v)
        if v_norm > 2.0:
            self.v = (self.v / v_norm) * 2.0

        # P 행렬 대각선 클램핑 (장시간 사용 시 무한 성장 → 갑작스런 튐 방지)
        # 상한은 충분히 넓게 잡아서 정상 동작에는 영향 없고, 10분+ 장시간 사용 시만 개입
        diag = np.diag(P)
        mask = diag > self._p_max
        if np.any(mask):
            idx = np.where(mask)[0]
            for i in idx:
                P[i, i] = self._p_max[i]
                
        self._sanitize_state()

    def detect_zupt(self, accel_th=0.05, gyro_th=0.05):
        if len(self.a_win) < self.window_size:
            return False
            
        a_var = np.var(self.a_win, axis=0)
        g_var = np.var(self.g_win, axis=0)
        
        # SHOE-like Energy Score
        a_mean = np.mean(self.a_win, axis=0)
        a_n = np.linalg.norm(a_mean)
        is_gravity = abs(a_n - 9.81) < 0.5
        
        a_score = np.sum(a_var)
        g_score = np.sum(g_var)
        
        # 임계값: 파라미터는 분산의 합 기준
        if a_score < (accel_th * 3) and g_score < (gyro_th * 3) and is_gravity:
            return True
        return False

    def update_zupt(self, current_gyro=None):
        # [ZARU] 정지 상태 자이로 바이어스 보정
        if current_gyro is not None:
            H_zaru = np.zeros((3, 15))
            H_zaru[0:3, 12:15] = np.eye(3)
            # z_g is the negative of the expected measurement
            z_g = current_gyro - self.w_b
            
            # pinv와 1e-9 오프셋 추가로 수치 안정성 확보
            inv_matrix = np.linalg.pinv(H_zaru @ self.P @ H_zaru.T + self.R_zaru + 1e-9 * np.eye(3))
            K_g = self.P @ H_zaru.T @ inv_matrix
            # [핵심] ZARU(자이로 바이어스 보정)가 회전각(q)을 미친듯이 꼬아버리는 참사 방지
            K_g[6:9, :] = 0
            dx_g = K_g @ z_g
            self._inject_error(dx_g)
            
        # [ZUPT] 정지 상태 속도 0 보정
        H_zupt = np.zeros((3, 15))
        H_zupt[0:3, 3:6] = np.eye(3)
        z_v = np.zeros(3) - self.v
        
        # pinv와 1e-9 오프셋 추가로 수치 안정성 확보
        inv_matrix_v = np.linalg.pinv(H_zupt @ self.P @ H_zupt.T + self.R_zupt + 1e-9 * np.eye(3))
        K_v = self.P @ H_zupt.T @ inv_matrix_v
        # [핵심] ZUPT가 속도를 강제로 0으로 깎을 때, 필터가 "앗 자세(q)가 잘못되었구나!" 하고 
        # 회전각을 강제로 비틀어버리는 현상(버튼 뗄 때 튀는 현상)을 방지합니다.
        K_v[6:9, :] = 0  
        dx_v = K_v @ z_v
        self._inject_error(dx_v)
        
    def update_mag(self, current_mag):
        """[Phase 5] 지자기 센서를 이용한 9축 헤딩(Yaw) 고정"""
        if self.mag_ref_world is None:
            return
            
        m_w = self.q.as_matrix() @ current_mag
        
        # World 수평면(XY) 투영
        m_w_xy = m_w[:2]
        n_w = np.linalg.norm(m_w_xy)
        if n_w < 1e-4: return
        m_w_xy /= n_w
        
        m_ref_xy = self.mag_ref_world[:2]
        n_ref = np.linalg.norm(m_ref_xy)
        if n_ref < 1e-4: return
        m_ref_xy /= n_ref
        
        # 측정된 북쪽과 레퍼런스 북쪽 간의 각도 오차 계산
        yaw_err = np.arctan2(m_w_xy[1], m_w_xy[0]) - np.arctan2(m_ref_xy[1], m_ref_xy[0])
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        
        # 요동침(Spike) 방지를 위해 아주 약한 게인(0.01)으로 보정 (Complementary 방식)
        # 장시간 길게 문장 쓸 때 서서히 방향이 틀어지는 드리프트만 원천 방지
        gain = 0.01
        correction = np.array([0, 0, yaw_err * gain])
        
        # World 기반 회전 적용!
        dq = Rotation.from_rotvec(correction)
        self.q = dq * self.q
        self._sanitize_state()

    def update_gravity_mahony(self, current_accel, alpha=0.002):
        """센서가 기울어질 때 발생하는 미세한 물리적 자이로 오차(Gravity-Bleed)를 보정하여 
        Pitch/Roll 위아래 드리프트를 영구적으로 차단합니다."""
        a_n = np.linalg.norm(current_accel)
        # 외부 가속이 심할 때(1G에서 멀어질 때)는 중력 보정을 스킵합니다
        if a_n < 9.0 or a_n > 12.0:
            return
            
        a_norm = current_accel / a_n
        
        # 센서 좌표계에서 예상되는 중력(Gravity) 방향
        g_sensor = self.q.inv().as_matrix() @ np.array([0.0, 0.0, 1.0])
        
        # 실제 가속도계가 가리키는 중력 방향과의 오차 (외적)
        err = np.cross(a_norm, g_sensor)
        
        # [핵심 수정] 자이로 바이어스(Drift) 자동 교정 로직 삭제
        # 이유: 손을 좌우로 흔들 때 발생하는 선형 가속도를 "센서가 기울어졌다"고 착각하고, 
        # 이에 대한 보상으로 자이로 센서 수치(w_b)를 왜곡시킵니다. 
        # 시간이 지날수록 축이 45도 이상 완전히 틀어져서, 좌우로 그어도 대각선으로 써지는 치명적 원인이 되었습니다.
        # 회전(q)를 중력 방향으로 아주 미세하게 당김
        if np.linalg.norm(err) > 1e-6:
            err_world = self.q.as_matrix() @ err
            dq = Rotation.from_rotvec(-err_world * alpha)
            self.q = dq * self.q
            self._sanitize_state()

    def _inject_error(self, dx):
        if np.isnan(dx).any() or np.isinf(dx).any():
            dx = np.zeros(15)
            
        self.p += dx[0:3]
        self.v += dx[3:6]
        
        angle = np.linalg.norm(dx[6:9])
        if angle > 1e-8:
            axis = dx[6:9] / angle
            dq = Rotation.from_rotvec(axis * angle)
            self.q = self.q * dq
            
        self.a_b += dx[9:12]
        self.w_b += dx[12:15]
        self._sanitize_state()

    def _sanitize_state(self):
        # Sanitize p
        if np.isnan(self.p).any() or np.isinf(self.p).any():
            self.p = np.zeros(3)
        # Sanitize v
        if np.isnan(self.v).any() or np.isinf(self.v).any():
            self.v = np.zeros(3)
        # Sanitize q
        q_val = self.q.as_quat()
        if np.isnan(q_val).any() or np.isinf(q_val).any() or np.linalg.norm(q_val) < 1e-6:
            self.q = Rotation.from_quat([0, 0, 0, 1])
        else:
            q_val = q_val / np.linalg.norm(q_val)
            self.q = Rotation.from_quat(q_val)
        # Sanitize a_b
        if np.isnan(self.a_b).any() or np.isinf(self.a_b).any():
            self.a_b = np.zeros(3)
        # Sanitize w_b
        if np.isnan(self.w_b).any() or np.isinf(self.w_b).any():
            self.w_b = np.zeros(3)
        # Sanitize P
        if np.isnan(self.P).any() or np.isinf(self.P).any():
            self.P[np.isnan(self.P)] = 0.0
            self.P[np.isinf(self.P)] = 0.0
            # Make sure diagonal is positive
            for i in range(15):
                if self.P[i, i] <= 0:
                    self.P[i, i] = 0.01
        np.clip(self.P, -1e6, 1e6, out=self.P)
