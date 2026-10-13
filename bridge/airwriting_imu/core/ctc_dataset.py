"""
CTC Dataset — CTC 학습을 위한 가변 길이 시퀀스 데이터셋
=======================================================

기존 GestureDataset과의 핵심 차이:
  - GestureDataset: 200프레임 고정 리샘플링 → 단일 라벨 (분류)
  - CTCDataset: 원본 길이 보존 + 패딩 → 문자열 라벨 (시퀀스 인식)

CTC 학습에서 시간축이 중요한 이유:
  리샘플링하면 "쓰는 속도" 정보가 사라져서 blank/반복 패턴을 학습할 수 없음.
  원본 길이를 유지해야 모델이 "이 구간은 아직 글자를 안 쓰고 있다(blank)" vs
  "이 구간에서 글자를 쓰고 있다(character)"를 구별할 수 있음.

두 가지 모드:
  1. single_char: 기존 단일 글자 JSON → 각각 1글자 CTC 학습
  2. word: 여러 글자 JSON을 랜덤 연결하여 합성 단어 생성 (데이터 증강)
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# 프로젝트 경로
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_DIR = _PROJECT_ROOT / "dataset"


class CTCDataset(Dataset):
    """
    CTC 학습용 데이터셋.
    
    각 샘플:
      - features: [T_i, 11] 가변 길이 특징 벡터 (패딩 전)
      - target: [S_i] 문자 인덱스 시퀀스 (A=1, B=2, ..., Z=26, blank=0)
      
    DataLoader 사용 시 반드시 CTCDataset.collate_fn을 사용해야 함:
      loader = DataLoader(dataset, batch_size=16, collate_fn=CTCDataset.collate_fn)
    """
    
    # 문자 → 인덱스 매핑 (CTC blank=0이므로 문자는 1부터)
    BLANK_IDX = 0
    CHAR_TO_IDX = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        mode: str = "single_char",       # "single_char" 또는 "word"
        word_length_range: Tuple[int, int] = (2, 5),  # word 모드 합성 단어 길이
        synth_samples: int = 500,         # word 모드 합성 샘플 수
        min_seq_len: int = 5,             # 최소 시퀀스 길이 (너무 짧은 것 제거)
        normalize: bool = True,           # StandardScaler 적용 여부
    ):
        if data_dir is None:
            data_dir = str(_DATASET_DIR)
        
        self.data_dir = data_dir
        self.mode = mode
        self.word_length_range = word_length_range
        self.synth_samples = synth_samples
        self.min_seq_len = min_seq_len
        self.normalize = normalize
        
        # 데이터 저장소
        self.samples: List[np.ndarray] = []     # 각 원소: [T_i, 11]
        self.targets: List[List[int]] = []      # 각 원소: [S_i]
        self.target_strings: List[str] = []     # 디버그용 원본 문자열
        
        # 글자별 원본 데이터 (word 합성용)
        self._char_pool: Dict[str, List[np.ndarray]] = {}
        
        self.scaler = StandardScaler() if normalize else None
        
        self._load_and_build()
    
    def _extract_features(self, strokes_list: list) -> np.ndarray:
        """Thin wrapper over ai_model.extract_features: float32 for CTC training."""
        from airwriting_imu.core.ai_model import extract_features
        arr = extract_features(strokes_list)
        if arr.size == 0:
            return np.zeros((0, 11), dtype=np.float32)
        return arr.astype(np.float32, copy=False)
    
    def _load_and_build(self):
        """데이터 로드 + 모드별 데이터셋 구성"""
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.json")))
        
        if not files:
            print(f"[CTCDataset] 데이터 없음: {self.data_dir}")
            return
        
        # Phase 1: 모든 JSON 로드 → 글자별 풀 구성
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                
                if isinstance(data, dict) and "strokes" in data:
                    strokes_list = data["strokes"]
                    label = data.get("label", "Unknown").upper()
                else:
                    strokes_list = [data]
                    label = os.path.basename(f).split('_')[0].upper()
                
                # 유효한 A-Z만 허용
                if label not in self.CHAR_TO_IDX:
                    continue
                
                # 원시 스트로크의 총 포인트 수 검증
                total_pts = sum(len(st) for st in strokes_list)
                if total_pts < self.min_seq_len:
                    continue
                
                if label not in self._char_pool:
                    self._char_pool[label] = []
                self._char_pool[label].append(strokes_list)
                
            except Exception as e:
                print(f"[CTCDataset] 로드 실패 {f}: {e}")
        
        available_chars = sorted(self._char_pool.keys())
        total_samples = sum(len(v) for v in self._char_pool.values())
        print(f"[CTCDataset] 로드 완료: {len(available_chars)}개 문자, {total_samples}개 샘플")
        print(f"[CTCDataset] 문자 분포: {', '.join(f'{c}:{len(self._char_pool[c])}' for c in available_chars)}")
        
        if not self._char_pool:
            return
        
        # Phase 2: 모드별 데이터셋 구성
        if self.mode == "single_char":
            self._build_single_char()
        elif self.mode == "word":
            self._build_single_char()  # 기본 단일 글자도 포함
            self._build_synthetic_words()  # + 합성 단어 추가
        
        # Phase 3: 정규화 (scaler)
        if self.normalize and len(self.samples) > 0:
            all_points = np.vstack(self.samples)
            self.scaler.fit(all_points)
            self.samples = [self.scaler.transform(s) for s in self.samples]
        
        print(f"[CTCDataset] 최종 데이터셋: {len(self.samples)}개 샘플 (모드: {self.mode})")
    
    def _build_single_char(self):
        """단일 글자 모드: 각 JSON = 1글자 CTC 샘플"""
        for char, sequences in self._char_pool.items():
            char_idx = self.CHAR_TO_IDX[char]
            for seq in sequences:
                features = self._extract_features(seq)
                if len(features) < self.min_seq_len:
                    continue
                self.samples.append(features)
                self.targets.append([char_idx])
                self.target_strings.append(char)
    
    def _build_synthetic_words(self):
        """
        합성 단어 모드: 실제 자주 쓰이는 영단어 사전을 기반으로 단어 생성.
        
        글자들을 원시 스트로크 상태로 먼저 이어 붙인 후,
        전체 단어에 대해 extract_features를 일괄 호출하여 피처 스케일링을 추론 환경과 일치시킵니다.
        """
        # 자주 쓰이는 2~5글자 기초 영단어 사전
        COMMON_WORDS = [
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN",
            "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
            "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO",
            "BOY", "DID", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "THAT", "WITH",
            "HAVE", "THIS", "WILL", "YOUR", "FROM", "THEY", "KNOW", "WANT", "BEEN",
            "GOOD", "MUCH", "SOME", "TIME", "VERY", "WHEN", "COME", "HERE", "JUST",
            "LIKE", "LONG", "MAKE", "MANY", "MORE", "ONLY", "OVER", "SUCH", "TAKE",
            "THAN", "THEM", "WELL", "WERE", "WHAT", "ABOUT", "WHICH", "THEIR", "OTHER",
            "COULD", "WHILE", "THESE", "THEN", "FIRST", "WATER", "AFTER", "WHERE", "RIGHT",
            "THINK", "THREE", "YEARS", "PLACE", "SOUND", "GREAT", "AGAIN", "STILL", "EVERY",
            "SMALL", "FOUND", "THOSE", "NEVER", "UNDER", "MIGHT", "WHILE", "HOUSE", "WORLD",
            "BELOW", "ASKED", "GOING", "LARGE", "UNTIL", "ALONG", "SHALL", "BEING", "OFTEN",
            "EARTH", "BEGAN", "SINCE", "STUDY", "NIGHT", "LIGHT", "ABOVE", "PAPER", "PARTS",
            "YOUNG", "STORY", "POINT", "TIMES", "HEARD", "WHOLE", "WHITE", "GIVEN", "MEANS",
            "APPLE", "SMART", "WATCH", "TEST", "HELLO", "YES", "NO", "RUN", "STOP"
        ]
        available_chars = list(self._char_pool.keys())
        if len(available_chars) < 2:
            print("[CTCDataset] 합성 단어 생성 불가: 2종류 이상의 글자 필요")
            return
            
        # 사용 가능한 글자로만 구성된 단어 필터링
        valid_words = []
        for w in COMMON_WORDS:
            if all(c in available_chars for c in w):
                valid_words.append(w)
                
        if not valid_words:
            print("[CTCDataset] 합성 단어 생성 불가: 조합 가능한 실제 단어가 없습니다.")
            return
        
        rng = np.random.RandomState(42)
        generated = 0
        
        for _ in range(self.synth_samples):
            # 실제 영단어 중 랜덤 선택
            word_str = valid_words[rng.randint(0, len(valid_words))]
            word_chars = list(word_str)
            
            # 각 글자의 원시 스트로크 리스트를 복사하여 합침
            word_strokes = []
            target_indices = []
            
            for char in word_chars:
                pool = self._char_pool[char]
                raw_strokes = pool[rng.randint(0, len(pool))]
                
                # 스트로크 리스트의 개별 스트로크(딕셔너리 리스트)를 복사하여 합침
                char_strokes = [list(st) for st in raw_strokes]
                word_strokes.extend(char_strokes)
                target_indices.append(self.CHAR_TO_IDX[char])
            
            # 전체 합쳐진 원시 스트로크들로부터 일괄 피처 추출 진행
            # 이 시점에 단어 전체 기준 바운딩 박스 스케일링 및 획 연속 계산이 수행됩니다.
            features = self._extract_features(word_strokes)
            
            if len(features) < self.min_seq_len:
                continue
            
            # 50% 확률로 데이터 증강(노이즈, 스케일링) 적용
            if rng.rand() > 0.5:
                features = self._augment_sequence(features, rng)
            
            self.samples.append(features)
            self.targets.append(target_indices)
            self.target_strings.append(word_str)
            generated += 1
        
        print(f"[CTCDataset] 합성 단어 {generated}개 생성 (전체 단어 피처 일괄 추출 + On-the-fly 증강 사용)")
    
    def _augment_sequence(self, seq: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        IMU 데이터 온더플라이 증강 (Time-Warping, Jittering, Scaling, Rotation, 2D Scaling)
        """
        aug_seq = seq.copy()
        
        # 0. Time-Warping (속도 왜곡)
        if rng.rand() > 0.5:
            T = len(aug_seq)
            sigma = 0.2
            warp = np.cumsum(np.abs(rng.normal(0, sigma, T)) + 1.0)
            warp = warp / warp[-1] * (T - 1)
            new_seq = np.zeros_like(aug_seq)
            for i in range(aug_seq.shape[1]):
                if i == 4:  # is_new_stroke
                    src = np.clip(np.round(np.interp(np.arange(T), warp, np.arange(T))).astype(int), 0, T - 1)
                    new_seq[:, i] = aug_seq[src, i]
                else:
                    new_seq[:, i] = np.interp(np.arange(T), warp, aug_seq[:, i])
            aug_seq = new_seq
        
        # 1. Jittering (Noise)
        noise_acc = rng.normal(0, 0.05, size=(aug_seq.shape[0], 3))
        noise_gyro = rng.normal(0, 0.02, size=(aug_seq.shape[0], 3))
        aug_seq[:, 5:8] += noise_acc
        aug_seq[:, 8:11] += noise_gyro
        
        # 2. IMU Scaling (0.9 ~ 1.1)
        scale = rng.uniform(0.9, 1.1)
        aug_seq[:, 5:11] *= scale
        
        # 3. 2D Trajectory Rotation (-20 to +20 degrees)
        if rng.rand() > 0.5:
            angle = np.radians(rng.uniform(-20, 20))
            c, s = np.cos(angle), np.sin(angle)
            x, y = aug_seq[:, 0].copy(), aug_seq[:, 1].copy()
            aug_seq[:, 0] = x * c - y * s
            aug_seq[:, 1] = x * s + y * c
            dx, dy = aug_seq[:, 2].copy(), aug_seq[:, 3].copy()
            aug_seq[:, 2] = dx * c - dy * s
            aug_seq[:, 3] = dx * s + dy * c
            
        # 4. 2D Trajectory Scaling (0.75 ~ 1.25, independent X and Y scaling)
        if rng.rand() > 0.5:
            sx = rng.uniform(0.75, 1.25)
            sy = rng.uniform(0.75, 1.25)
            aug_seq[:, 0] *= sx
            aug_seq[:, 1] *= sy
            aug_seq[:, 2] *= sx
            aug_seq[:, 3] *= sy
        
        return aug_seq

    def _generate_gap_frames(self, num_frames: int, last_point: Optional[np.ndarray] = None) -> np.ndarray:
        """
        글자 사이의 "비필기 구간" 프레임 생성.
        
        정지 상태를 시뮬레이션: 위치 유지, 가속도 ≈ 중력, 자이로 ≈ 0
        이 구간에서 CTC는 blank를 출력하도록 학습됨.
        """
        gap = np.zeros((num_frames, 11), dtype=np.float32)
        
        if last_point is not None:
            # 마지막 위치 유지 (x, y)
            gap[:, 0] = last_point[0]  # x
            gap[:, 1] = last_point[1]  # y
        
        # dx, dy = 0 (이동 없음)
        # is_new_stroke = 0
        
        # 정지 가속도 (중력): az ≈ 9.81 + 작은 노이즈
        gap[:, 7] = 9.81 + np.random.randn(num_frames) * 0.05  # az
        gap[:, 5] = np.random.randn(num_frames) * 0.1  # ax (노이즈)
        gap[:, 6] = np.random.randn(num_frames) * 0.1  # ay (노이즈)
        
        # 자이로 ≈ 0 (정지)
        gap[:, 8:11] = np.random.randn(num_frames, 3) * 0.01
        
        return gap
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            features: [T_i, 11] — 가변 길이 시퀀스
            target: [S_i] — 타겟 문자 인덱스
            input_length: 시퀀스 길이
            target_length: 타겟 문자열 길이
        """
        # X, Y 절대 좌표(0번, 1번 인덱스)를 잘라내어 11차원 -> 9차원으로 축소
        # 텔레포트 현상을 방지하고 순수 운동 법칙(속도, 가속도, 각속도)에만 의존하게 함
        seq_trimmed = self.samples[idx][:, 2:]
        features = torch.tensor(seq_trimmed, dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        
        return features, target, len(features), len(target)
    
    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        가변 길이 배치를 패딩하여 텐서로 변환.
        
        DataLoader에 반드시 이 함수를 전달해야 함:
          loader = DataLoader(dataset, collate_fn=CTCDataset.collate_fn)
        
        Returns:
            features_padded: [B, T_max, 11] — 제로 패딩된 배치
            targets: [sum(S_i)] — 1D concatenated 타겟 (CTC 표준 포맷)
            input_lengths: [B] — 각 샘플의 원본 시퀀스 길이
            target_lengths: [B] — 각 샘플의 타겟 문자열 길이
        """
        features_list, targets_list, input_lengths, target_lengths = zip(*batch)
        
        # 시퀀스 패딩 (길이 맞추기)
        features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
        
        # 타겟은 1D로 concat (CTC 표준)
        targets_concat = torch.cat(targets_list)
        
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        return features_padded, targets_concat, input_lengths, target_lengths
    
    def get_vocab_size(self) -> int:
        """CTC 모델에 전달할 vocab 크기 (blank 포함)"""
        return len(self.CHAR_TO_IDX) + 1  # +1 for blank
    
    def get_stats(self) -> dict:
        """데이터셋 통계"""
        if not self.samples:
            return {"total": 0}
        
        lengths = [len(s) for s in self.samples]
        target_lens = [len(t) for t in self.targets]
        
        return {
            "total_samples": len(self.samples),
            "available_chars": sorted(self._char_pool.keys()),
            "num_chars": len(self._char_pool),
            "seq_length": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(np.mean(lengths), 1),
                "median": round(np.median(lengths), 1),
            },
            "target_length": {
                "min": min(target_lens),
                "max": max(target_lens),
                "mean": round(np.mean(target_lens), 1),
            },
            "mode": self.mode,
        }


# ═══════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print("  CTCDataset — CTC 학습용 가변 길이 데이터셋")
    print("=" * 60)
    
    # 1. 단일 글자 모드 테스트
    print("\n--- 모드 1: single_char ---")
    ds_single = CTCDataset(mode="single_char")
    stats = ds_single.get_stats()
    print(f"   통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    if len(ds_single) > 0:
        # DataLoader 테스트
        loader = DataLoader(ds_single, batch_size=8, shuffle=True, collate_fn=CTCDataset.collate_fn)
        batch = next(iter(loader))
        features, targets, input_lengths, target_lengths = batch
        print(f"\n   배치 테스트:")
        print(f"   features: {features.shape}")
        print(f"   targets: {targets.shape} (concat)")
        print(f"   input_lengths: {input_lengths.tolist()}")
        print(f"   target_lengths: {target_lengths.tolist()}")
    
    # 2. 합성 단어 모드 테스트
    print("\n--- 모드 2: word (합성 단어) ---")
    ds_word = CTCDataset(mode="word", synth_samples=100, word_length_range=(2, 4))
    stats_word = ds_word.get_stats()
    print(f"   통계: {json.dumps(stats_word, indent=2, ensure_ascii=False)}")
    
    if len(ds_word) > 0:
        # 합성 단어 샘플 확인
        print(f"\n   합성 단어 예시:")
        for i in range(min(10, len(ds_word))):
            if len(ds_word.target_strings[i]) > 1:
                print(f"   [{i}] '{ds_word.target_strings[i]}' — {len(ds_word.samples[i])} frames")
    
    print(f"\n✅ CTCDataset 검증 완료!")
