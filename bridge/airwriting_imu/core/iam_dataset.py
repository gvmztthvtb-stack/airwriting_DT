"""
JW v1 — IAM Handwriting Dataset Loader
========================================
IAM Handwriting Database (word-level)를 JW v1 학습 포맷으로 변환.

지원 모드:
  1. word_image: IAM 단어 이미지 → 128×128 리사이즈 → 분류/인식 학습
  2. char_image: IAM 단어 이미지에서 개별 문자 크롭 → 글자 인식 학습
  3. hybrid: 자체 에어라이팅 데이터 + IAM 데이터 혼합 학습

디렉토리 구조 (Kaggle에서 다운로드 후 배치):
  data/iam/
    words/           ← IAM words 이미지 (a01/a01-000/a01-000-00-00.png)
    words.txt        ← IAM 라벨 파일
    
Kaggle: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
115,320개 영어 단어 이미지 (657명 필기자, 300dpi)
"""

import os
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class IAMDatasetLoader:
    """IAM Handwriting Database word-level 로더"""
    
    def __init__(self, iam_root: str = "data/iam", image_size: int = 128):
        """
        Args:
            iam_root: IAM 데이터 루트 경로
            image_size: 출력 이미지 크기 (정사각형)
        """
        self.iam_root = Path(iam_root)
        self.image_size = image_size
        
        # Kaggle 데이터셋 자동 경로 탐색
        # 구조 1: data/iam/words.txt + data/iam/words/
        # 구조 2: data/iam/iam_words/words.txt + data/iam/iam_words/words/  (Kaggle)
        # 구조 3: data/iam/words_new.txt + data/iam/iam_words/words/
        self.words_file = None
        self.words_dir = None
        
        # 우선순위 탐색
        candidates = [
            (self.iam_root / "iam_words" / "words.txt", self.iam_root / "iam_words" / "words"),
            (self.iam_root / "words.txt", self.iam_root / "words"),
            (self.iam_root / "words_new.txt", self.iam_root / "iam_words" / "words"),
            (self.iam_root / "words_new.txt", self.iam_root / "words"),
        ]
        
        for wf, wd in candidates:
            if wf.exists() and wd.exists():
                self.words_file = wf
                self.words_dir = wd
                break
        
        # 폴백
        if self.words_file is None:
            self.words_file = self.iam_root / "words.txt"
            self.words_dir = self.iam_root / "words"
        
        self._entries = []
        self._label_map = {}
    
    def is_available(self) -> bool:
        """IAM 데이터셋이 다운로드되어 있는지 확인"""
        return self.words_file.exists() and self.words_dir.exists()
    
    def get_status(self) -> dict:
        """데이터셋 상태 리포트"""
        if not self.is_available():
            return {
                "available": False,
                "message": f"IAM 데이터를 {self.iam_root}에 배치하세요.",
                "download_url": "https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database",
                "required_files": ["words.txt", "words/ (이미지 폴더)"],
            }
        
        if not self._entries:
            self.parse_labels()
        
        return {
            "available": True,
            "total_words": len(self._entries),
            "unique_labels": len(set(e[1] for e in self._entries)),
            "root": str(self.iam_root),
        }
    
    def parse_labels(self, max_word_len: int = 10, min_samples: int = 3):
        """
        words.txt 파싱.
        
        Args:
            max_word_len: 최대 단어 길이 (짧은 영어 단어만)
            min_samples: 최소 샘플 수 (너무 적은 레이블 제외)
        """
        if not self.words_file.exists():
            print(f"⚠️ {self.words_file} 없음")
            return
        
        entries = []
        label_count = {}
        
        with open(self.words_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(' ')
                if len(parts) < 9:
                    continue
                
                # parts[1] = segmentation result (ok/err)
                if parts[1] != 'ok':
                    continue
                
                word_id = parts[0]     # e.g., a01-000u-00-00
                transcription = parts[-1]  # 실제 단어
                
                # 필터: 영문+숫자만, 최대 길이 제한
                if not transcription.isascii():
                    continue
                if len(transcription) > max_word_len:
                    continue
                if len(transcription) < 1:
                    continue
                
                # 이미지 경로 구성
                id_parts = word_id.split('-')
                img_path = self.words_dir / id_parts[0] / f"{id_parts[0]}-{id_parts[1]}" / f"{word_id}.png"
                
                if img_path.exists():
                    entries.append((str(img_path), transcription))
                    label_count[transcription] = label_count.get(transcription, 0) + 1
        
        # 최소 샘플 수 필터
        if min_samples > 1:
            valid_labels = {k for k, v in label_count.items() if v >= min_samples}
            entries = [(p, l) for p, l in entries if l in valid_labels]
        
        self._entries = entries
        print(f"✅ IAM 로드: {len(entries)} words, {len(set(e[1] for e in entries))} unique labels")
    
    def load_images(self, mode: str = "word", max_samples: int = None,
                    target_labels: list = None) -> list:
        """
        이미지 로드 + 전처리.
        
        Args:
            mode: "word" (전체 단어) | "char" (개별 글자)
            max_samples: 최대 로드 수
            target_labels: 특정 라벨만 로드 (예: ["hello", "world"])
        
        Returns:
            list of {"image": np.ndarray [H,W], "label": str}
        """
        if not HAS_PIL:
            print("⚠️ Pillow 필요: pip install Pillow")
            return []
        
        if not self._entries:
            self.parse_labels()
        
        results = []
        entries = self._entries
        
        if target_labels:
            target_set = set(target_labels)
            entries = [(p, l) for p, l in entries if l in target_set]
        
        if max_samples and len(entries) > max_samples:
            # 랜덤 샘플링
            rng = np.random.RandomState(42)
            indices = rng.choice(len(entries), max_samples, replace=False)
            entries = [entries[i] for i in indices]
        
        for img_path, label in entries:
            try:
                img = Image.open(img_path).convert('L')  # 그레이스케일
                
                if mode == "char":
                    # 개별 글자 크롭 (단순 등분)
                    w, h = img.size
                    char_w = w // len(label) if len(label) > 0 else w
                    for ci, ch in enumerate(label):
                        x1 = ci * char_w
                        x2 = min((ci + 1) * char_w, w)
                        char_img = img.crop((x1, 0, x2, h))
                        char_arr = self._preprocess(char_img)
                        results.append({
                            "image": char_arr,
                            "label": ch,
                            "source": "iam_char",
                        })
                else:
                    # 전체 단어 이미지
                    arr = self._preprocess(img)
                    results.append({
                        "image": arr,
                        "label": label,
                        "source": "iam_word",
                    })
            except Exception as e:
                continue
        
        print(f"📦 IAM 이미지 로드: {len(results)}개 ({mode} mode)")
        return results
    
    def load_characters(self, max_per_char: int = 200) -> list:
        """
        A-Z, a-z, 0-9 개별 문자 인식용 데이터 로드.
        각 문자당 최대 max_per_char개.
        """
        if not self._entries:
            self.parse_labels()
        
        char_items = {}
        
        for img_path, label in self._entries:
            for ch in label:
                if ch.isalnum():
                    if ch not in char_items:
                        char_items[ch] = []
                    if len(char_items[ch]) < max_per_char:
                        char_items[ch].append((img_path, label))
        
        results = self.load_images(mode="char", 
                                   target_labels=list(set(l for _, l in self._entries)))
        
        # 글자별 균형 맞추기
        char_groups = {}
        for r in results:
            ch = r["label"]
            if ch not in char_groups:
                char_groups[ch] = []
            if len(char_groups[ch]) < max_per_char:
                char_groups[ch].append(r)
        
        balanced = []
        for ch, items in sorted(char_groups.items()):
            balanced.extend(items)
        
        print(f"🔤 글자별 데이터: {len(char_groups)}종, 총 {len(balanced)}개")
        return balanced
    
    def _preprocess(self, img: Image.Image) -> np.ndarray:
        """
        이미지 전처리: 종횡비 유지 리사이즈 + 패딩 → 정사각형.
        
        배경 흰색 → 검정으로 반전 (에어라이팅 데이터와 동일하게).
        """
        sz = self.image_size
        
        # 배경 반전 (IAM: 흰 배경+검은 글씨 → 검은 배경+흰 글씨)
        img = ImageOps.invert(img)
        
        # 종횡비 유지 리사이즈
        w, h = img.size
        scale = min(sz / w, sz / h) * 0.85  # 85%만 채움 (여백)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 정사각형 캔버스에 중앙 배치
        canvas = Image.new('L', (sz, sz), 0)
        x_offset = (sz - new_w) // 2
        y_offset = (sz - new_h) // 2
        canvas.paste(img, (x_offset, y_offset))
        
        return np.array(canvas, dtype=np.float32) / 255.0
    
    def create_hybrid_dataset(self, airwriting_dir: str = "dataset",
                              iam_mode: str = "word",
                              max_iam: int = 1000) -> list:
        """
        자체 에어라이팅 데이터 + IAM 데이터 혼합.
        
        에어라이팅 데이터를 우선시하고, IAM은 보조 학습용.
        """
        from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
        
        results = []
        
        # 1. 자체 에어라이팅 데이터
        renderer = TrajectoryRenderer(size=self.image_size)
        airwriting = renderer.render_dataset(airwriting_dir)
        for item in airwriting:
            item["source"] = "airwriting"
        results.extend(airwriting)
        print(f"✍️ 에어라이팅: {len(airwriting)}개")
        
        # 2. IAM 데이터
        if self.is_available():
            iam = self.load_images(mode=iam_mode, max_samples=max_iam)
            results.extend(iam)
            print(f"📚 IAM: {len(iam)}개")
        else:
            print("⚠️ IAM 데이터 없음, 에어라이팅 데이터만 사용")
        
        print(f"📊 총 하이브리드 데이터: {len(results)}개")
        return results
    
    def download_instructions(self) -> str:
        """IAM 데이터셋 다운로드 안내"""
        return """
═══════════════════════════════════════════════════
  IAM Handwriting Database 설치 가이드
═══════════════════════════════════════════════════

방법 1: Kaggle에서 다운로드 (추천)
  1. https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
  2. [Download] 버튼 클릭 (1.1GB ZIP)
  3. 압축 해제 후 data/iam/ 폴더에 배치

방법 2: Kaggle CLI 자동 다운로드
  kaggle datasets download -d nibinv23/iam-handwriting-word-database -p data/iam --unzip

최종 디렉토리 구조:
  data/iam/
    words.txt          ← 라벨 파일 (115,320개 단어)
    words/             ← words/ 폴더 전체
      a01/
        a01-000/
          a01-000-00-00.png
          ...

확인:
  python -c "from airwriting_imu.core.iam_dataset import IAMDatasetLoader; print(IAMDatasetLoader().get_status())"

═══════════════════════════════════════════════════
"""


# ─── Kaggle 간이 다운로더 (대안) ───

def try_kaggle_download(dest: str = "data/iam"):
    """
    Kaggle에서 IAM 데이터셋 다운로드 시도.
    kaggle API 키가 설정되어 있어야 합니다.
    https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
    """
    try:
        os.makedirs(dest, exist_ok=True)
        import subprocess
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d',
             'nibinv23/iam-handwriting-word-database', '-p', dest, '--unzip'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ Kaggle에서 IAM 다운로드 완료: {dest}")
            return True
        else:
            print(f"⚠️ Kaggle CLI 오류: {result.stderr}")
            return False
    except FileNotFoundError:
        print("⚠️ Kaggle CLI 미설치. pip install kaggle 후 API 키를 설정하세요.")
        print("  또는 수동 다운로드: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database")
        return False
    except Exception as e:
        print(f"⚠️ Kaggle 다운로드 실패: {e}")
        return False


if __name__ == "__main__":
    loader = IAMDatasetLoader()
    
    status = loader.get_status()
    print(f"\n📊 IAM Status: {status}")
    
    if not status["available"]:
        print(loader.download_instructions())
    else:
        # 테스트 로드
        items = loader.load_images(mode="word", max_samples=10)
        for it in items[:5]:
            print(f"  label='{it['label']}' shape={it['image'].shape}")
        
        # 하이브리드 테스트
        hybrid = loader.create_hybrid_dataset(max_iam=50)
        sources = {}
        for h in hybrid:
            s = h.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
        print(f"\n하이브리드 소스: {sources}")
