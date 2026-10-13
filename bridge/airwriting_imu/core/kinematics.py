import numpy as np

class KinematicChain:
    def __init__(self, max_reach_s1_s2=0.25, max_reach_s2_s3=0.18, max_reach_s1_s3=0.51):
        """
        S1 (손목/전완근) -> S2 (손등) -> S3 (검지손가락)
        """
        self.max_reach_s1_s2 = max_reach_s1_s2
        self.max_reach_s2_s3 = max_reach_s2_s3
        self.max_reach_s1_s3 = max_reach_s1_s3

    def clamp_distance(self, root_p, target_p, max_dist):
        """root_p로부터 target_p까지의 거리가 max_dist를 넘으면 강제로 끌어당김"""
        diff = target_p - root_p
        dist = np.linalg.norm(diff)
        if dist > max_dist:
            # 방향은 유지하되, 최대 거리로 잘라냄
            corrected_target = root_p + (diff / dist) * max_dist
            return corrected_target, True
        return target_p, False

    def unocclude_drift(self, p1, p2, p3):
        """
        S1, S2, S3의 ESKF 위치를 입력받아 해부학적 허용치 밖으로 벗어난 위치를 교정 (Loop Closure)
        Returns:
            corrected_p1, corrected_p2, corrected_p3, (bool: is_clamped)
        """
        clamped_any = False
        
        # 1. S1 -> S2 클램핑
        cp2, clamped = self.clamp_distance(p1, p2, self.max_reach_s1_s2)
        clamped_any = clamped_any or clamped

        # 2. S2 -> S3 클램핑
        cp3, clamped = self.clamp_distance(cp2, p3, self.max_reach_s2_s3)
        clamped_any = clamped_any or clamped

        # 3. S1 -> S3 절대 클램핑 (안전장치)
        cp3, clamped = self.clamp_distance(p1, cp3, self.max_reach_s1_s3)
        clamped_any = clamped_any or clamped
        
        return p1, cp2, cp3, clamped_any
