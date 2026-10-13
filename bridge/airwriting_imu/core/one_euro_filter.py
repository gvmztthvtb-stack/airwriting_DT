"""
One Euro Filter — 속도 적응형 저지연 스무딩 필터

논문: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
      Géry Casiez, Nicolas Roussel, Daniel Vogel (CHI 2012)

핵심 원리:
  - 느린 움직임 → 낮은 cutoff → 강한 스무딩 (떨림/드리프트 제거)
  - 빠른 움직임 → 높은 cutoff → 약한 스무딩 (밀림/지연 제거)

파라미터 튜닝 가이드:
  1. beta=0으로 시작, min_cutoff를 낮춰서 정지 시 떨림 제거
  2. beta를 올려서 빠른 움직임 시 밀림 제거
"""

import math


class LowPassFilter:
    """단순 1차 저역통과 필터 (지수 이동 평균의 일반화)"""
    
    def __init__(self, alpha=1.0):
        self.y = None
        self.s = None
        self.set_alpha(alpha)
    
    def set_alpha(self, alpha):
        alpha = max(0.0, min(1.0, alpha))
        self.a = alpha
    
    def filter(self, value):
        if self.y is None:
            self.s = value
        else:
            self.s = self.a * value + (1.0 - self.a) * self.s
        self.y = value
        return self.s
    
    def has_last_value(self):
        return self.y is not None
    
    def last_value(self):
        return self.y


class OneEuroFilter:
    """
    One Euro Filter: 속도 적응형 스무딩.
    
    Args:
        freq:       센서 샘플링 주파수 (Hz). 예: 100Hz
        min_cutoff: 최소 cutoff 주파수 (Hz). 낮을수록 정지 시 안정적. 기본 1.0
        beta:       속도-cutoff 비례 계수. 높을수록 빠른 움직임 추종력 ↑. 기본 0.007
        d_cutoff:   미분(속도) 신호의 cutoff 주파수 (Hz). 기본 1.0
    """
    
    def __init__(self, freq=100.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None
    
    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, x, timestamp=None):
        # 타임스탬프 기반 동적 주파수 계산
        if self.last_time is not None and timestamp is not None:
            dt = timestamp - self.last_time
            if dt > 1e-6:
                self.freq = 1.0 / dt
        self.last_time = timestamp
        
        # 1. 속도(미분) 추정
        if self.x_filter.has_last_value():
            dx = (x - self.x_filter.last_value()) * self.freq
        else:
            dx = 0.0
        
        # 2. 속도 신호를 스무딩
        edx = self.dx_filter.filter(dx)
        self.dx_filter.set_alpha(self._alpha(self.d_cutoff))
        
        # 3. 속도에 비례하여 cutoff 주파수 증가
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # 4. 적응형 alpha로 원본 신호 스무딩
        self.x_filter.set_alpha(self._alpha(cutoff))
        return self.x_filter.filter(x)
    
    def reset(self):
        """필터 상태 초기화 (캘리브레이션 리셋 시 호출)"""
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None
