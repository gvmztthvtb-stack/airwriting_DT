"""
JW v1 — Trajectory Renderer
============================
디지털트윈 궤적 데이터를 128×128 흑백 이미지로 변환.
AI 모델의 Image Branch 입력용.
"""

import numpy as np
import json
import os
import glob

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️ Pillow 미설치. trajectory_renderer 이미지 렌더링 비활성화.")
    print("  설치: py -3 -m pip install Pillow")


class TrajectoryRenderer:
    """디지털트윈 궤적 JSON → 128×128 그레이스케일 이미지 변환기"""
    
    def __init__(self, size: int = 128, line_width: int = 3, 
                 padding: float = 0.1, blur_radius: float = 0.8):
        self.size = size
        self.line_width = line_width
        self.padding = padding
        self.blur_radius = blur_radius
    
    def render(self, strokes: list, size: int = None) -> np.ndarray:
        """
        strokes → (size, size) float32 numpy array [0.0 ~ 1.0]
        
        Args:
            strokes: list of list of dicts with 'x', 'y' keys
            size: override image size
        Returns:
            np.ndarray shape (H, W) dtype float32
        """
        sz = size or self.size
        
        if not HAS_PIL:
            return np.zeros((sz, sz), dtype=np.float32)
        
        # 모든 포인트 수집
        all_x, all_y = [], []
        for stroke in strokes:
            for pt in stroke:
                all_x.append(pt.get('x', 0.0))
                all_y.append(pt.get('y', 0.0))
        
        if len(all_x) < 2:
            return np.zeros((sz, sz), dtype=np.float32)
        
        # 바운딩 박스 계산 + 패딩
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        range_x = max_x - min_x if max_x > min_x else 1.0
        range_y = max_y - min_y if max_y > min_y else 1.0
        
        # 종횡비 유지하면서 정사각형에 맞추기
        max_range = max(range_x, range_y)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        pad = self.padding * max_range
        half = max_range / 2 + pad
        
        # 좌표 → 픽셀 변환 함수
        def to_pixel(x, y):
            px = int((x - cx + half) / (2 * half) * (sz - 1))
            py = int((cy - y + half) / (2 * half) * (sz - 1))  # Y축 반전
            px = max(0, min(sz - 1, px))
            py = max(0, min(sz - 1, py))
            return px, py
        
        # PIL로 안티앨리어싱 렌더링
        img = Image.new('L', (sz, sz), 0)
        draw = ImageDraw.Draw(img)
        
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            
            points = []
            for pt in stroke:
                px, py = to_pixel(pt.get('x', 0), pt.get('y', 0))
                points.append((px, py))
            
            # 연속된 점 쌍으로 선분 그리기
            for i in range(len(points) - 1):
                draw.line([points[i], points[i+1]], fill=255, width=self.line_width)
        
        # 약간의 가우시안 블러 (노이즈 방지 + 안티앨리어싱)
        if self.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        # numpy 변환 (0~1 정규화)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    
    def render_from_json(self, filepath: str) -> tuple:
        """
        JSON 파일 → (이미지, 라벨)
        Returns: (np.ndarray, str)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "strokes" in data:
            strokes = data["strokes"]
            label = data.get("label", "Unknown")
        else:
            strokes = [data] if isinstance(data, list) else []
            label = os.path.basename(filepath).split('_')[0]
        
        img = self.render(strokes)
        return img, label
    
    def render_dataset(self, data_dir: str = "dataset") -> list:
        """
        dataset 폴더 전체 → [(이미지, 라벨, 파일명), ...]
        """
        results = []
        files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        for f in files:
            try:
                img, label = self.render_from_json(f)
                results.append({
                    "image": img,
                    "label": label,
                    "file": os.path.basename(f),
                    "path": f,
                })
            except Exception as e:
                print(f"⚠️ 렌더링 실패 {f}: {e}")
        
        return results
    
    def save_preview(self, strokes: list, path: str):
        """디버깅용: 궤적을 이미지 파일로 저장"""
        if not HAS_PIL:
            return
        arr = self.render(strokes)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
        img.save(path)


if __name__ == "__main__":
    renderer = TrajectoryRenderer(size=128)
    results = renderer.render_dataset("dataset")
    print(f"렌더링 완료: {len(results)}개 샘플")
    for r in results[:5]:
        print(f"  {r['file']:40s} label={r['label']:6s} shape={r['image'].shape}")
