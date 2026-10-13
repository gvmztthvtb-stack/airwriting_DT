"""
JW v1 — Data Augmentor
=======================
소량 데이터(33개)를 수천 개로 증강하는 파이프라인.
궤적(좌표) + IMU(시계열) 동시 증강.
"""

import numpy as np
import copy


class DataAugmentor:
    """실제 에어라이팅 데이터를 기반으로 합성 변형 생성"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def augment_sample(self, strokes: list, n_augments: int = 10) -> list:
        """
        원본 1개 → n_augments개의 변형 생성.
        
        Args:
            strokes: list of list of dicts (원본 궤적)
            n_augments: 생성할 변형 수
        Returns:
            list of augmented stroke sets
        """
        augmented = []
        transforms = [
            self._jitter, self._scale, self._rotate,
            self._translate, self._speed_variation,
            self._elastic_distortion, self._stroke_dropout,
        ]
        
        for _ in range(n_augments):
            new_strokes = copy.deepcopy(strokes)
            
            # 랜덤하게 1~3개 변환 조합 적용
            n_transforms = self.rng.randint(1, 4)
            selected = self.rng.choice(len(transforms), n_transforms, replace=False)
            
            for idx in selected:
                new_strokes = transforms[idx](new_strokes)
            
            augmented.append(new_strokes)
        
        return augmented
    
    def _jitter(self, strokes: list) -> list:
        """각 점에 가우시안 노이즈 추가 (±2px)"""
        sigma = self.rng.uniform(0.5, 3.0)
        for stroke in strokes:
            for pt in stroke:
                pt['x'] = pt.get('x', 0) + self.rng.normal(0, sigma)
                pt['y'] = pt.get('y', 0) + self.rng.normal(0, sigma)
                # IMU도 약간의 노이즈
                for key in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                    if key in pt:
                        pt[key] += self.rng.normal(0, 0.05)
        return strokes
    
    def _scale(self, strokes: list) -> list:
        """전체 궤적 크기 변환 (0.7~1.3배) 및 가속도 물리량 동시 비례 조정"""
        factor = self.rng.uniform(0.7, 1.3)
        cx, cy = self._center(strokes)
        for stroke in strokes:
            for pt in stroke:
                pt['x'] = cx + (pt.get('x', 0) - cx) * factor
                pt['y'] = cy + (pt.get('y', 0) - cy) * factor
                
                # [Jetson 패치] 거리 스케일에 비례하여 2D 물리 가속도 크기 스케일링
                if 'ax' in pt: pt['ax'] = pt['ax'] * factor
                if 'ay' in pt: pt['ay'] = pt['ay'] * factor
        return strokes
    
    def _rotate(self, strokes: list) -> list:
        """전체 궤적 회전 (-15° ~ +15°) 및 2D 가속도/각속도 벡터 동시 회전 변환"""
        angle = self.rng.uniform(-15, 15) * np.pi / 180
        cx, cy = self._center(strokes)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for stroke in strokes:
            for pt in stroke:
                # 1) 좌표 회전
                dx = pt.get('x', 0) - cx
                dy = pt.get('y', 0) - cy
                pt['x'] = cx + dx * cos_a - dy * sin_a
                pt['y'] = cy + dx * sin_a + dy * cos_a
                
                # [Jetson 패치] 2) 2D 가속도 벡터 (ax, ay) 동시 회전
                ax, ay = pt.get('ax', 0.0), pt.get('ay', 0.0)
                pt['ax'] = ax * cos_a - ay * sin_a
                pt['ay'] = ax * sin_a + ay * cos_a
                
                # [Jetson 패치] 3) 2D 각속도 벡터 (gx, gy) 동시 회전
                gx, gy = pt.get('gx', 0.0), pt.get('gy', 0.0)
                pt['gx'] = gx * cos_a - gy * sin_a
                pt['gy'] = gx * sin_a + gy * cos_a
        return strokes
    
    def _translate(self, strokes: list) -> list:
        """전체 궤적 이동 (±10%)"""
        all_x = [pt.get('x', 0) for s in strokes for pt in s]
        all_y = [pt.get('y', 0) for s in strokes for pt in s]
        if not all_x:
            return strokes
        rng = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)
        dx = self.rng.uniform(-0.1, 0.1) * rng
        dy = self.rng.uniform(-0.1, 0.1) * rng
        for stroke in strokes:
            for pt in stroke:
                pt['x'] = pt.get('x', 0) + dx
                pt['y'] = pt.get('y', 0) + dy
        return strokes
    
    def _speed_variation(self, strokes: list) -> list:
        """시간(dt) 변형 및 가속도/자이로 각속도 동역학 비례 동기화"""
        factor = self.rng.uniform(0.6, 1.5)
        inv_f = 1.0 / factor
        inv_f2 = 1.0 / (factor * factor)
        
        for stroke in strokes:
            for pt in stroke:
                if 'dt' in pt:
                    pt['dt'] *= factor
                # [Jetson 패치] 1) 가속도는 dt 제곱에 반비례하여 세기 조절
                for k in ['ax', 'ay', 'az']:
                    if k in pt: pt[k] *= inv_f2
                # [Jetson 패치] 2) 각속도는 dt에 반비례하여 세기 조절
                for k in ['gx', 'gy', 'gz']:
                    if k in pt: pt[k] *= inv_f
        return strokes
    
    def _elastic_distortion(self, strokes: list) -> list:
        """탄성 변형 — 자연스러운 필기 변형 시뮬레이션"""
        alpha = self.rng.uniform(2.0, 8.0)
        sigma = self.rng.uniform(1.0, 3.0)
        
        for stroke in strokes:
            n = len(stroke)
            if n < 3:
                continue
            # 저주파 노이즈 생성 (가우시안 필터링 효과)
            dx_field = self.rng.normal(0, 1, n)
            dy_field = self.rng.normal(0, 1, n)
            
            # 간이 가우시안 스무딩
            kernel_size = max(3, int(sigma * 2) | 1)
            kernel = np.exp(-np.arange(-kernel_size//2, kernel_size//2+1)**2 / (2*sigma**2))
            kernel /= kernel.sum()
            
            if n > kernel_size:
                dx_field = np.convolve(dx_field, kernel, mode='same')
                dy_field = np.convolve(dy_field, kernel, mode='same')
            
            for i, pt in enumerate(stroke):
                pt['x'] = pt.get('x', 0) + dx_field[i] * alpha
                pt['y'] = pt.get('y', 0) + dy_field[i] * alpha
        
        return strokes
    
    def _stroke_dropout(self, strokes: list) -> list:
        """획 내 일부 포인트 드롭 (센서 노이즈 시뮬레이션)"""
        if len(strokes) == 0:
            return strokes
        new_strokes = []
        for stroke in strokes:
            if len(stroke) < 5:
                new_strokes.append(stroke)
                continue
            # 10~20% 포인트 드롭
            drop_rate = self.rng.uniform(0.05, 0.15)
            kept = [pt for pt in stroke if self.rng.random() > drop_rate]
            if len(kept) > 3:
                new_strokes.append(kept)
            else:
                new_strokes.append(stroke)
        return new_strokes
    
    def _center(self, strokes: list) -> tuple:
        """궤적 중심점 계산"""
        all_x = [pt.get('x', 0) for s in strokes for pt in s]
        all_y = [pt.get('y', 0) for s in strokes for pt in s]
        if not all_x:
            return 0.0, 0.0
        return np.mean(all_x), np.mean(all_y)


if __name__ == "__main__":
    # 테스트: 더미 데이터로 증강
    dummy = [[{"x": i*10, "y": i*5, "ax": 0.1, "dt": 0.01} for i in range(20)]]
    aug = DataAugmentor()
    results = aug.augment_sample(dummy, n_augments=5)
    print(f"원본 1개 → {len(results)}개 증강 완료")
    for i, r in enumerate(results):
        pts = r[0]
        print(f"  변형 {i}: {len(pts)} points, "
              f"x_range=[{min(p['x'] for p in pts):.1f}, {max(p['x'] for p in pts):.1f}]")
