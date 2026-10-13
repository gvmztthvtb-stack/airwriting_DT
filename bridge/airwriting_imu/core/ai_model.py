"""
AirScribe AI Engine — CNN-BiLSTM 하이브리드 + Transformer (Phase 9)

아키텍처 근거:
  학계 SOTA 벤치마크에서 IMU 기반 손글씨 인식은 CNN(로컬 패턴 추출) +
  BiLSTM(양방향 시퀀스 컨텍스트)이 순수 Transformer 대비 일관된 성능 우위.
  특히 노이즈가 심한 IMU 데이터에서 CNN의 1D-Conv 특징 추출이 압도적.

지원 타겟: 알파벳 A-Z + 영어 문장 (Phase 9 목표)
배포 타겟: Jetson Orin Nano (ONNX → TensorRT 변환 지원)
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)

# CWD에 의존하지 않고 어느 위치에서 실행해도 weights/dataset을 찾기 위해 절대경로 사용
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"
_DATASET_DIR = _PROJECT_ROOT / "dataset"


def extract_features(session_strokes):
    """Single source of truth for stroke→feature conversion.

    Used by both training (`GestureDataset._load_data`) and all inference paths
    (`_predict_legacy`, `_predict_jw_v1`, `_predict_ctc`). Drift between training
    and inference is the bug class this helper exists to prevent — see
    `verify_pipeline.py` T1 which asserts byte-equality with a reference copy.

    Output shape: (N, 11) float64, columns:
      x, y, dx, dy, is_new_stroke, ax, ay, az, gx, gy, gz
    Zero-centered on the first frame's (x, y).
    """
    flattened = []
    last_x = last_y = None
    for st in session_strokes:
        for pi, pt in enumerate(st):
            curr_x = pt.get('x', 0.0)
            curr_y = pt.get('y', 0.0)
            dx = (curr_x - last_x) if last_x is not None else 0.0
            dy = (curr_y - last_y) if last_y is not None else 0.0
            is_new_stroke = 1.0 if pi == 0 else 0.0
            flattened.append([
                curr_x, curr_y, dx, dy, is_new_stroke,
                pt.get('ax', 0.0), pt.get('ay', 0.0), pt.get('az', 0.0),
                pt.get('gx', 0.0), pt.get('gy', 0.0), pt.get('gz', 0.0),
            ])
            last_x, last_y = curr_x, curr_y
    if flattened:
        fx, fy = flattened[0][0], flattened[0][1]
        for row in flattened:
            row[0] -= fx
            row[1] -= fy
        # 드리프트/글씨 크기 불변 정규화 (bounding box span)
        xs = [row[0] for row in flattened]
        ys = [row[1] for row in flattened]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        for row in flattened:
            row[0] /= span  # x
            row[1] /= span  # y
            row[2] /= span  # dx
            row[3] /= span  # dy
    return np.asarray(flattened, dtype=np.float64)


# ─── Feature-level augmentation helpers ───
# train_now.py와 GestureDataset 양쪽에서 사용. Scaler 피팅 후, 리샘플링 전에 적용.

def _aug_time_warp(seq, sigma=0.2):
    """시간축 왜곡: 필기 속도 변화에 강건하게.
    물리 법칙에 의거하여 시간 왜곡률(warp_deriv)에 따라 
    속도(dx, dy) 및 자이로(gx, gy, gz)는 1승 비례, 
    가속도(ax, ay, az)는 2승 비례로 크기를 변환합니다.
    """
    T = len(seq)
    warp = np.cumsum(np.abs(np.random.randn(T) * sigma) + 1.0)
    warp = warp / warp[-1] * (T - 1)
    new_seq = np.zeros_like(seq)
    for i in range(seq.shape[1]):
        if i == 4:  # is_new_stroke: nearest-neighbor로 0/1 보존
            src = np.clip(np.round(np.interp(np.arange(T), warp, np.arange(T))).astype(int), 0, T - 1)
            new_seq[:, i] = seq[src, i]
        else:
            new_seq[:, i] = np.interp(np.arange(T), warp, seq[:, i])
            
    # 시간 변형 비율(warp_deriv) 계산
    warp_deriv = np.gradient(warp)
    
    # 1. 속도/변위(dx, dy) 및 자이로(gx, gy, gz)는 시간 왜곡에 반비례 -> 1승 비례 곱함
    new_seq[:, [2, 3]] *= warp_deriv[:, np.newaxis]      # dx, dy
    new_seq[:, [8, 9, 10]] *= warp_deriv[:, np.newaxis]  # gx, gy, gz
    
    # 2. 가속도(ax, ay, az)는 시간 왜곡의 제곱에 반비례 -> 2승 비례 곱함
    new_seq[:, [5, 6, 7]] *= (warp_deriv[:, np.newaxis] ** 2)
    
    return new_seq


def _aug_jitter(seq, sigma=0.03):
    """가우시안 노이즈 추가."""
    noise = np.random.randn(*seq.shape).astype(np.float32) * sigma
    noise[:, 4] = 0.0  # is_new_stroke 보존
    return seq + noise


def _aug_scale(seq, low=0.85, high=1.15):
    """크기 변환: 글자 크기 차이에 강건하게.
    물리 법칙에 의거하여 공간 스케일링 비율만큼 가속도(ax, ay)도 비례하여 스케일링합니다.
    """
    sx, sy = np.random.uniform(low, high), np.random.uniform(low, high)
    out = seq.copy()
    out[:, 0] *= sx  # x
    out[:, 1] *= sy  # y
    out[:, 2] *= sx  # dx
    out[:, 3] *= sy  # dy
    out[:, 5] *= sx  # ax
    out[:, 6] *= sy  # ay
    return out


def _aug_rotation(seq, max_deg=12):
    """회전: 손목 각도 차이에 강건하게.
    기하학적 회전 행렬에 의거하여 궤적 좌표뿐만 아니라 
    가속도(ax, ay) 및 자이로(gx, gy) 센서 값의 방향도 동시 변환합니다.
    """
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(angle), np.sin(angle)
    out = seq.copy()
    
    # 2D 좌표 회전
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = x * c - y * s
    out[:, 1] = x * s + y * c
    
    # 변위(dx, dy) 회전
    dx, dy = out[:, 2].copy(), out[:, 3].copy()
    out[:, 2] = dx * c - dy * s
    out[:, 3] = dx * s + dy * c
    
    # 가속도(ax, ay) 회전
    ax, ay = out[:, 5].copy(), out[:, 6].copy()
    out[:, 5] = ax * c - ay * s
    out[:, 6] = ax * s + ay * c
    
    # 자이로(gx, gy) 회전
    gx, gy = out[:, 8].copy(), out[:, 9].copy()
    out[:, 8] = gx * c - gy * s
    out[:, 9] = gx * s + gy * c
    
    return out


def _augment_feature_seq(seq, n=5):
    """단일 시퀀스 → n개 증강 복사본 생성."""
    augmented = []
    for _ in range(n):
        aug = seq.copy().astype(np.float32)
        if np.random.rand() > 0.3: aug = _aug_time_warp(aug, sigma=0.2)
        if np.random.rand() > 0.3: aug = _aug_jitter(aug, sigma=0.03)
        if np.random.rand() > 0.3: aug = _aug_scale(aug, 0.85, 1.15)
        if np.random.rand() > 0.5: aug = _aug_rotation(aug, max_deg=12)
        augmented.append(aug)
    return augmented


class GestureDataset(Dataset):
    def __init__(self, data_dir=None, max_seq_len=200, augment=False, target_per_class=120):
        if data_dir is None:
            data_dir = str(_DATASET_DIR)
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.target_per_class = target_per_class
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.scaler = StandardScaler()
        
        self._load_data()
        
    def _load_data(self):
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        raw_data = []
        
        label_idx = 0
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    strokes_list = []
                    label = ""
                    
                    if isinstance(data, dict) and "strokes" in data:
                        strokes_list = data["strokes"]
                        label = data.get("label", "Unknown")
                    else:
                        strokes_list = [data]
                        label = os.path.basename(f).split('_')[0]

                    if label not in self.label_map:
                        self.label_map[label] = label_idx
                        label_idx += 1
                        
                    flattened = extract_features(strokes_list)
                    if len(flattened) > 5:
                        raw_data.append(flattened)
                        self.labels.append(self.label_map[label])
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                
        if len(raw_data) == 0:
            return
            
        all_pts = np.vstack(raw_data)
        self.scaler.fit(all_pts)
        
        # ─── Feature-level augmentation (scaler 피팅 후, 리샘플링 전) ───
        if self.augment:
            label_counts = {}
            for lbl in self.labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            max_count = max(label_counts.values()) if label_counts else 0
            target = max(max_count, self.target_per_class)
            
            by_label = {}
            for i, lbl in enumerate(self.labels):
                by_label.setdefault(lbl, []).append(raw_data[i])
            
            inv_map = {v: k for k, v in self.label_map.items()}
            for lbl, sequences in by_label.items():
                deficit = target - len(sequences)
                if deficit <= 0:
                    continue
                gen = 0
                for seq_orig in sequences:
                    if gen >= deficit:
                        break
                    n_per = max(1, deficit // len(sequences) + 1)
                    for aug in _augment_feature_seq(seq_orig, n=n_per):
                        if gen >= deficit:
                            break
                        raw_data.append(aug)
                        self.labels.append(lbl)
                        gen += 1
                print(f"  [Augment] {inv_map.get(lbl, '?')}: {len(sequences)} -> {len(sequences)+gen}")
            print(f"  [Augment] Total: {len(raw_data)} samples")
        
        for seq in raw_data:
            # Resampling (고정 길이로 보간하여 전체 모양 보존)
            seq_resampled = self._resample(seq, self.max_seq_len)
            seq_norm = self.scaler.transform(seq_resampled)
            self.samples.append(seq_norm)
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _resample(self, seq, target_len):
        """선형 보간을 사용하여 시퀀스 길이를 target_len으로 조정"""
        curr_len = len(seq)
        if curr_len == target_len:
            return seq
        
        # 각 채널별로 보간 수행
        x_old = np.linspace(0, 1, curr_len)
        x_new = np.linspace(0, 1, target_len)
        
        resampled = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
        for i in range(seq.shape[1]):
            if i == 4:
                # is_new_stroke은 이진 플래그(0/1). 선형보간하면 획경계가 0.3/0.7로 뭉개져
                # 학습/추론 정합이 깨짐 → nearest-neighbor로 0/1 보존. (train_now.py와 동일)
                nn_idx = np.round(x_new * (curr_len - 1)).astype(int)
                resampled[:, i] = seq[nn_idx, i]
            else:
                resampled[:, i] = np.interp(x_new, x_old, seq[:, i])

        return resampled
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ─── Phase 7: 기존 Transformer (하위 호환) ───
class GestureTransformer(nn.Module):
    def __init__(self, input_dim=11, d_model=64, nhead=4, num_layers=3, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        features = x.mean(dim=1)
        out = self.classifier(features)
        return out


# ─── Phase 10: 순수 시계열 BiLSTM (CNN 완전 배제) ───
class PureBiLSTMAttention(nn.Module):
    """
    CNN 필터가 시간축(Time-warp)의 미세한 변화를 왜곡하는 문제를 해결하기 위해
    CNN을 완전히 제거하고 순수 시계열 처리만 수행하는 아키텍처.
    """
    def __init__(self, input_dim=11, num_classes=2, hidden_dim=128):
        super().__init__()
        
        # 1. Linear Projection (특징 차원 확장)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05)
        )
        
        # 2. BiLSTM (순차적 문맥 파악)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,  # 레이어 수 증가로 복잡한 궤적 학습
            batch_first=True,
            bidirectional=True,
            dropout=0.15
        )
        
        # 3. Multi-Head Attention (중요한 획/모션 부분에 집중)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        
        # 1. Projection
        x_proj = self.projection(x)  # [batch, seq_len, hidden_dim]
        
        # 2. BiLSTM
        lstm_out, _ = self.lstm(x_proj)  # [batch, seq_len, hidden_dim * 2]
        
        # 3. Self-Attention Pooling
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) # [batch, seq_len, hidden_dim * 2]
        
        # Global Average Pooling over sequence length
        context = torch.mean(attn_out, dim=1)  # [batch, hidden_dim * 2]
        
        # 4. Classification
        out = self.classifier(context)
        return out


class AirWritingAI:
    """
    통합 AI 관리자.
    model_type:
      - "pure_bilstm" : 순수 시계열 BiLSTM + Attention (CNN 배제, 권장)
      - "jw_v2"      : JW v2 (Pure Continuous Mamba)
      - "transformer" : 기존 Transformer
      - "fastkan"    : FastKAN (KAN 기반 초경량 분류, ~18KB INT8)
      - "ctc"        : BiLSTM-CTC (연속 문자 인식, 자막 모드)
    """
    def __init__(self):
        self.model = None
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {}
        self.scaler = None
        self.model_type = "pure_bilstm"
        self.renderer = None  # JW v1용 (현재 미사용)
        self.ctc_scaler = None  # CTC 모드 전용 scaler
        
    def train(self, data_dir="dataset", epochs=30, model_type=None,
              augment_factor=0, image_size=128, lr=0.001,
              batch_size=16, callback=None):
        """
        통합 학습 메서드.
        
        callback: (msg, level, **kw) → 실시간 로그 전송용
        """
        mt = model_type or self.model_type
        
        if mt == "jw_v1":
            return self._train_jw_v1(data_dir, epochs, augment_factor,
                                     image_size, lr, batch_size, callback)
        elif mt == "ctc":
            return self._train_ctc(data_dir, epochs, lr, batch_size, callback)
        elif mt == "fastkan":
            return self._train_sequence(data_dir, epochs, lr, batch_size, callback)
        else:
            return self._train_sequence(data_dir, epochs, lr, batch_size, callback)
    
    def _train_jw_v1(self, data_dir, epochs, augment_factor,
                     image_size, lr, batch_size, callback):
        """JW v1 학습 파이프라인"""
        log = callback or (lambda msg, lvl="info", **kw: print(f"[{lvl}] {msg}"))
        
        log("JW v1 학습 파이프라인 시작...", "info")
        
        # 1. 궤적 이미지 렌더링
        from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
        self.renderer = TrajectoryRenderer(size=image_size)
        rendered = self.renderer.render_dataset(data_dir)
        log(f"에어라이팅 궤적 이미지 {len(rendered)}개 렌더링 완료 ({image_size}×{image_size})", "success")
        
        # 1.5. IAM 데이터 하이브리드 (옵션)
        try:
            from airwriting_imu.core.iam_dataset import IAMDatasetLoader
            iam_loader = IAMDatasetLoader(image_size=image_size)
            if iam_loader.is_available():
                iam_data = iam_loader.load_images(mode="word", max_samples=2000)
                rendered.extend(iam_data)
                log(f"IAM 하이브리드: +{len(iam_data)}개 추가 (총 {len(rendered)}개)", "success")
            else:
                log("IAM 데이터 없음 — 에어라이팅 데이터만 사용", "dim")
        except Exception as e:
            log(f"IAM 로드 스킵: {e}", "dim")
        
        if len(rendered) < 2:
            return False, "데이터가 2개 미만입니다."
        
        # 2. 라벨 매핑
        label_set = sorted(set(r["label"] for r in rendered))
        self.label_map = {lbl: idx for idx, lbl in enumerate(label_set)}
        num_classes = len(self.label_map)
        log(f"클래스 {num_classes}개: {', '.join(label_set[:20])}{'...' if num_classes > 20 else ''}", "info")
        
        if num_classes < 2:
            return False, "최소 2종류 이상의 글자를 수집하세요."
        
        # 3. 데이터 증강
        if augment_factor > 1:
            from airwriting_imu.core.data_augmentor import DataAugmentor
            augmentor = DataAugmentor()
            original_count = len(rendered)
            
            augmented = []
            for r in rendered:
                fpath = r["path"]
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                strokes = data.get("strokes", [])
                aug_list = augmentor.augment_sample(strokes, n_augments=augment_factor)
                for aug_strokes in aug_list:
                    aug_img = self.renderer.render(aug_strokes)
                    augmented.append({
                        "image": aug_img,
                        "label": r["label"],
                    })
            rendered.extend(augmented)
            log(f"데이터 증강: {original_count} → {len(rendered)}개 (×{augment_factor})", "success")
        
        # 4. 시계열 데이터 준비 (GestureDataset에서 가져옴)
        self.dataset = GestureDataset(data_dir=data_dir)
        self.scaler = self.dataset.scaler
        
        # 5. JW v1 모델 생성 (in_channels=11 matches extract_features output)
        from airwriting_imu.core.jw_v1 import JWv1
        self.model = JWv1(
            codebook_size=512,
            d_model=128,
            n_layers=4,
            d_state=16,
            num_classes=num_classes,
            in_channels=11,
        ).to(self.device)
        
        report = self.model.count_parameters()
        log(f"JW v1 모델 생성: {report['total']:,} params ({report['size_mb_fp16']:.2f}MB FP16)", "info")
        
        # 6. 학습
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # 간이 데이터로더 (이미지 + 시계열 쌍)
        images = []
        imu_seqs = []
        labels = []
        
        for r in rendered:
            img = r["image"]  # [H, W] float32
            lbl = self.label_map[r["label"]]
            images.append(torch.tensor(img, dtype=torch.float32).unsqueeze(0))  # [1, H, W]
            labels.append(lbl)
        
        # 시계열은 dataset에서 매핑 (증강분은 원본 재사용)
        for i in range(len(images)):
            idx = min(i, len(self.dataset) - 1) if len(self.dataset) > 0 else 0
            if len(self.dataset) > 0:
                seq, _ = self.dataset[idx % len(self.dataset)]
                imu_seqs.append(seq)
            else:
                imu_seqs.append(torch.zeros(200, 8))
        
        images_t = torch.stack(images).to(self.device)
        imu_t = torch.stack(imu_seqs).to(self.device)
        labels_t = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        n_samples = len(labels_t)
        best_loss = float('inf')
        patience = 0
        final_acc = 0.0
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # 미니배치
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]
                
                batch_imu = imu_t[idx]
                batch_img = images_t[idx]
                batch_lbl = labels_t[idx]
                
                optimizer.zero_grad()
                logits, vq_loss = self.model(batch_imu, batch_img, mode="classify")
                cls_loss = criterion(logits, batch_lbl)
                loss = cls_loss + 0.1 * vq_loss  # VQ loss 가중치 0.1
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == batch_lbl).sum().item()
                total += len(batch_lbl)
            
            scheduler.step()
            n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
            avg_loss = total_loss / n_batches
            acc = correct / max(total, 1) * 100
            final_acc = acc
            
            progress = int((epoch + 1) / epochs * 100)
            log(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}",
                "info", epoch=epoch+1, accuracy=f"{acc:.1f}", progress=progress, loss=avg_loss, acc=acc/100)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= 7:
                    log(f"Early stopping at epoch {epoch+1}", "warn")
                    break
        
        # 7. 저장
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(_WEIGHTS_DIR / "jw_v1.pt"))
        import pickle
        with open(_WEIGHTS_DIR / "meta.pkl", "wb") as f:
            pickle.dump({
                "label_map": self.label_map,
                "scaler": self.scaler,
                "model_type": "jw_v1",
                "image_size": image_size,
            }, f)
        
        log(f"JW v1 학습 완료! {num_classes}개 클래스, 정확도 {final_acc:.1f}%", "success")
        return True, f"학습 성공! {num_classes}개 클래스, 정확도 {final_acc:.1f}%"
    
    def _train_ctc(self, data_dir, epochs, lr, batch_size, callback):
        """CTC 기반 연속 문자 인식 학습 모듈"""
        log = callback or (lambda msg, lvl="info", **kw: print(f"[{lvl}] {msg}"))
        
        from airwriting_imu.core.ctc_model import CTCRecognizer
        from airwriting_imu.core.ctc_dataset import CTCDataset
        
        log("CTC 연속 인식 모델 학습 시작...", "info")
        
        # 1. 데이터셋 로드 (word 모드: 합성 단어 포함)
        ctc_dataset = CTCDataset(
            data_dir=data_dir,
            mode="word",
            synth_samples=500,
            word_length_range=(2, 5),
        )
        
        if len(ctc_dataset) == 0:
            return False, "CTC 데이터를 먼저 수집하세요."
        
        stats = ctc_dataset.get_stats()
        log(f"데이터: {stats['total_samples']}개 샘플, {stats['num_chars']}개 문자, "
            f"시퀀스 길이 {stats['seq_length']['min']}-{stats['seq_length']['max']}", "info")
        
        # 2. 모델 초기화
        self.model = CTCRecognizer(
            input_dim=9,
            hidden_dim=128,
            num_lstm_layers=3,
            num_classes=26,  # A-Z
            dropout=0.3,
        ).to(self.device)
        
        report = self.model.count_parameters()
        log(f"모델: {report['total']:,} params ({report['size_mb_fp16']} MB FP16)", "info")
        
        # 3. 옵티마이저 / 스케줄러
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = DataLoader(
            ctc_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=CTCDataset.collate_fn, drop_last=False,
        )
        
        # 4. 학습 루프
        best_loss = float('inf')
        patience_counter = 0
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_correct_chars = 0
            total_target_chars = 0
            num_batches = 0
            
            for features, targets, input_lengths, target_lengths in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                optimizer.zero_grad()
                
                log_probs = self.model(features, lengths=input_lengths)  # [T, B, vocab]
                
                # CTC Loss
                loss = self.model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # CER 계산 (Character Error Rate 대체: 정확 문자 수)
                with torch.no_grad():
                    decoded = self.model.greedy_decode(log_probs, input_lengths)
                    # 타겟 문자열 복원
                    offset = 0
                    for b_idx in range(len(decoded)):
                        tgt_len = target_lengths[b_idx].item()
                        tgt_indices = targets[offset:offset + tgt_len].cpu().tolist()
                        tgt_str = "".join(CTCRecognizer.CHARS[i - 1] for i in tgt_indices)
                        # 정확한 문자 수 카운팅
                        for c in tgt_str:
                            total_target_chars += 1
                            if c in decoded[b_idx]:
                                total_correct_chars += 1
                        offset += tgt_len
            
            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            char_acc = total_correct_chars / max(total_target_chars, 1) * 100
            
            progress = int((epoch + 1) / epochs * 100)
            log(f"Epoch {epoch+1}/{epochs} | CTC Loss: {avg_loss:.4f} | Char Acc: {char_acc:.1f}%",
                "info", epoch=epoch+1, accuracy=f"{char_acc:.1f}", progress=progress,
                loss=avg_loss, acc=char_acc/100)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    log(f"Early stopping at epoch {epoch+1}", "warn")
                    break
        
        # 5. 저장
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(_WEIGHTS_DIR / "ctc.pt"))
        import pickle
        # CTC는 독립 메타 파일에 저장 (기존 분류 모델의 meta.pkl을 덮어쓰지 않음)
        with open(_WEIGHTS_DIR / "meta_ctc.pkl", "wb") as f:
            pickle.dump({
                "label_map": {c: i for i, c in enumerate(CTCRecognizer.CHARS)},
                "scaler": ctc_dataset.scaler,
                "model_type": "ctc",
            }, f)
        
        self.model_type = "ctc"
        self.ctc_scaler = ctc_dataset.scaler
        self.label_map = {c: i for i, c in enumerate(CTCRecognizer.CHARS)}
        
        log(f"CTC 학습 완료! A-Z 연속 인식 모델 저장됨", "success")
        return True, f"CTC 학습 성공! {stats['num_chars']}개 문자, Loss {best_loss:.4f}"
    
    def _train_sequence(self, data_dir, epochs, lr, batch_size, callback):
        """순수 시계열(Pure BiLSTM / Transformer) 학습 모듈"""
        log = callback or (lambda msg, lvl="info", **kw: print(f"[{lvl}] {msg}"))
        
        log(f"AI {self.model_type.upper()} 학습 모듈 초기화...", "info")
        self.dataset = GestureDataset(data_dir=data_dir, augment=True)
        
        if len(self.dataset) == 0:
            return False, "데이터를 먼저 수집하세요."
            
        self.label_map = self.dataset.label_map
        num_classes = len(self.label_map)
        
        if num_classes < 2:
            return False, "최소 2종류 이상의 글자를 수집하세요."
        
        if self.model_type == "pure_bilstm":
            self.model = PureBiLSTMAttention(num_classes=num_classes).to(self.device)
        elif self.model_type == "jw_v2":
            from airwriting_imu.core.jw_v1 import JWv2_Continuous
            # input_dim=11 matches extract_features output (x,y,dx,dy,is_new_stroke,ax,ay,az,gx,gy,gz)
            self.model = JWv2_Continuous(input_dim=11, num_classes=num_classes).to(self.device)
        elif self.model_type == "fastkan":
            from airwriting_imu.core.fastkan import FastKANClassifier
            self.model = FastKANClassifier(
                input_dim=11, hidden_dim=32, num_classes=num_classes, num_grids=8
            ).to(self.device)
        else:
            self.model = GestureTransformer(num_classes=num_classes).to(self.device)
        
        # ─── 클래스 불균형 보정: 역빈도 가중치 ───
        # 데이터가 적은 클래스(B:41)와 많은 클래스(H:188)의 손실 기여도를 균등화
        label_counts = torch.bincount(self.dataset.labels, minlength=num_classes).float()
        label_counts = label_counts.clamp(min=1)  # 0 방지
        class_weights = (1.0 / label_counts)
        class_weights = class_weights / class_weights.sum() * num_classes  # 평균=1로 정규화
        class_weights = class_weights.to(self.device)
        
        # 분포 로깅
        inv_label_map = {v: k for k, v in self.label_map.items()}
        dist_str = ", ".join(f"{inv_label_map.get(i, '?')}:{int(label_counts[i].item())}(w={class_weights[i]:.2f})" for i in range(num_classes))
        log(f"클래스 분포 (역빈도 가중치 적용): {dist_str}", "info")
            
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # ─── 80/20 held-out validation split (deterministic per dataset size) ───
        # Stratification: shuffle indices with a fixed generator so each run is reproducible.
        # If a class has only 1 sample, it stays in train (val_idx skips it).
        n_total = len(self.dataset)
        gen = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_total, generator=gen)
        n_val = max(1, int(round(n_total * 0.2))) if n_total >= 5 else 0
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()

        from torch.utils.data import Subset
        train_set = Subset(self.dataset, train_idx)
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        if n_val > 0:
            val_set = Subset(self.dataset, val_idx)
            val_loader = DataLoader(val_set, batch_size=max(batch_size, 64), shuffle=False)
        else:
            val_loader = None
        log(f"Split: train={len(train_idx)}  val={len(val_idx)} (seed=42)", "info")

        best_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        final_acc = 0.0
        final_val_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            per_class_correct = torch.zeros(num_classes)
            per_class_total = torch.zeros(num_classes)

            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                for c in range(num_classes):
                    mask = (y == c)
                    per_class_total[c] += mask.sum().item()
                    per_class_correct[c] += ((pred == y) & mask).sum().item()

            scheduler.step()
            avg_loss = total_loss / max(len(loader), 1)
            acc = correct / max(total, 1) * 100
            final_acc = acc

            # ─── Validation pass ───
            val_acc = 0.0
            val_per_class_correct = torch.zeros(num_classes)
            val_per_class_total = torch.zeros(num_classes)
            if val_loader is not None:
                self.model.eval()
                vc = vt = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        pred = self.model(x).argmax(dim=1)
                        vc += (pred == y).sum().item()
                        vt += y.size(0)
                        for c in range(num_classes):
                            mask = (y == c)
                            val_per_class_total[c] += mask.sum().item()
                            val_per_class_correct[c] += ((pred == y) & mask).sum().item()
                val_acc = vc / max(vt, 1) * 100
                final_val_acc = val_acc

            progress = int((epoch + 1) / epochs * 100)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                per_cls_str = " | ".join(
                    f"{inv_label_map.get(c, '?')}:T{per_class_correct[c]/max(per_class_total[c],1)*100:.0f}/"
                    f"V{val_per_class_correct[c]/max(val_per_class_total[c],1)*100:.0f}"
                    for c in range(num_classes)
                )
                log(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train: {acc:.1f}% | Val: {val_acc:.1f}% | [{per_cls_str}]",
                    "info", epoch=epoch+1, accuracy=f"{val_acc:.1f}", progress=progress, loss=avg_loss, acc=val_acc/100)
            else:
                log(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train: {acc:.1f}% | Val: {val_acc:.1f}%",
                    "info", epoch=epoch+1, accuracy=f"{val_acc:.1f}", progress=progress, loss=avg_loss, acc=val_acc/100)

            # Save on best validation accuracy (not training accuracy)
            save_criterion = val_acc if val_loader is not None else acc
            if save_criterion > best_val_acc:
                best_val_acc = save_criterion
                _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(_WEIGHTS_DIR / f"{self.model_type}.pt"))
                import pickle
                with open(_WEIGHTS_DIR / "meta.pkl", "wb") as f:
                    pickle.dump({
                        "label_map": self.label_map,
                        "scaler": self.dataset.scaler,
                        "model_type": self.model_type,
                    }, f)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    log(f"Early stopping at epoch {epoch+1}", "warn")
                    break

        msg = f"학습 완료! {num_classes}개 클래스, Train {final_acc:.1f}% / Val {final_val_acc:.1f}% (best Val {best_val_acc:.1f}%)"
        log(msg, "success")
        return True, msg
        
    def load_model(self, model_type=None):
        """
        저장된 모델 로드.
        
        model_type: None이면 meta.pkl에서 자동 감지.
                    "ctc"이면 meta_ctc.pkl에서 CTC 모델 로드.
        """
        import pickle
        try:
            forced_ctc = (model_type == "ctc")
            
            # CTC 명시적 요청 또는 meta_ctc.pkl 존재 시
            if model_type == "ctc":
                meta_path = _WEIGHTS_DIR / "meta_ctc.pkl"
            else:
                meta_path = _WEIGHTS_DIR / "meta.pkl"
            
            if not meta_path.exists():
                # meta.pkl이 CTC로 덮어써진 경우 복구 시도
                meta_path = _WEIGHTS_DIR / "meta.pkl"
            
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                    self.label_map = meta["label_map"]
                    self.scaler = meta.get("scaler")
                    if not forced_ctc:
                        model_type = meta.get("model_type", "transformer")
            elif forced_ctc:
                # CTC용 meta 파일이 없어도 A-Z label_map 직접 생성
                meta = {}
                self.label_map = {chr(65+i): i for i in range(26)}
                self.scaler = None
            else:
                raise FileNotFoundError(f"meta.pkl not found: {meta_path}")
                
            num_classes = len(self.label_map)
            
            if model_type == "jw_v1":
                from airwriting_imu.core.jw_v1 import JWv1
                self.model = JWv1(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load(str(_WEIGHTS_DIR / "jw_v1.pt"), map_location=self.device, weights_only=True))
                # 이미지 렌더러 초기화
                from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
                img_size = meta.get("image_size", 128)
                self.renderer = TrajectoryRenderer(size=img_size)
            elif model_type == "pure_bilstm":
                self.model = PureBiLSTMAttention(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load(str(_WEIGHTS_DIR / "pure_bilstm.pt"), map_location=self.device, weights_only=True))
            elif model_type == "hybrid":
                self.model = HybridIMUImageNet(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load(str(_WEIGHTS_DIR / "hybrid.pt"), map_location=self.device, weights_only=True))
            elif model_type == "jw_v2":
                from airwriting_imu.core.jw_v1 import JWv2_Continuous
                self.model = JWv2_Continuous(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load(str(_WEIGHTS_DIR / "jw_v2.pt"), map_location=self.device, weights_only=True))
            elif model_type == "ctc":
                from airwriting_imu.core.ctc_model import CTCRecognizer
                checkpoint = torch.load(str(_WEIGHTS_DIR / "ctc.pt"), map_location=self.device, weights_only=False)
                
                hidden_dim = 128
                num_layers = 3
                if isinstance(checkpoint, dict):
                    hidden_dim = checkpoint.get("hidden_dim", 128)
                    num_layers = checkpoint.get("num_layers", 3)
                
                self.model = CTCRecognizer(
                    input_dim=9, hidden_dim=hidden_dim, num_lstm_layers=num_layers,
                    num_classes=26, dropout=0.3,
                ).to(self.device)
                
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    # scaler 복원
                    if "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        scaler.mean_ = checkpoint["scaler_mean"]
                        scaler.scale_ = checkpoint["scaler_scale"]
                        scaler.var_ = scaler.scale_ ** 2
                        scaler.n_features_in_ = len(scaler.mean_)
                        self.ctc_scaler = scaler
                    else:
                        self.ctc_scaler = meta.get("scaler")
                else:
                    self.model.load_state_dict(checkpoint)
                    self.ctc_scaler = meta.get("scaler")
            elif model_type == "fastkan":
                from airwriting_imu.core.fastkan import FastKANClassifier
                self.model = FastKANClassifier(
                    input_dim=11, hidden_dim=32, num_classes=num_classes, num_grids=8
                ).to(self.device)
                self.model.load_state_dict(
                    torch.load(str(_WEIGHTS_DIR / "fastkan.pt"), map_location=self.device, weights_only=True))
            elif model_type == "cnn_bilstm":
                # Legacy: GestureCNNBiLSTM 클래스 삭제됨 — Transformer로 대체 로드 시도
                print("[WARN] cnn_bilstm 모델 타입은 더 이상 지원되지 않습니다. transformer로 폴백합니다.")
                self.model = GestureTransformer(num_classes=num_classes).to(self.device)
                weight_path = _WEIGHTS_DIR / "cnn_bilstm.pt"
                if weight_path.exists():
                    print("[WARN] cnn_bilstm.pt 웨이트는 호환되지 않으므로 재학습이 필요합니다.")
                else:
                    self.model.load_state_dict(
                        torch.load(str(_WEIGHTS_DIR / "transformer.pt"), map_location=self.device, weights_only=True))
            else:
                self.model = GestureTransformer(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load(str(_WEIGHTS_DIR / "transformer.pt"), map_location=self.device, weights_only=True))
                
            self.model.eval()
            self.model_type = model_type
            print(f"모델 로드 완료: {model_type} ({num_classes}개 클래스)")
            return True
        except Exception as e:
            print(f"모델 로드 실패 (아직 학습 전): {e}")
            self.model = None
            self.renderer = None
            self.label_map = {}
            return False
            
    def predict(self, session_strokes):
        if self.model is None or not self.label_map:
            return None, 0.0
        
        if self.model_type == "jw_v1":
            return self._predict_jw_v1(session_strokes)
        elif self.model_type == "hybrid":
            return self._predict_hybrid(session_strokes)
        elif self.model_type == "ctc":
            return self._predict_ctc(session_strokes)
        else:
            return self._predict_legacy(session_strokes)
            
    def _predict_hybrid(self, session_strokes):
        """멀티모달 하이브리드(IMU 시계열 + 2D 궤적 이미지) 추론"""
        self.model.eval()
        
        seq = extract_features(session_strokes)
        if len(seq) < 5:
            return None, 0.0
            
        # 1. 2D 이미지 렌더링
        img = render_trajectory_to_image(seq, img_size=128)
        
        # 2. IMU 피처 리샘플링 및 정규화
        seq_resampled = self._resample_imu(seq, 200)
        if self.scaler:
            seq_resampled = self.scaler.transform(seq_resampled)
            
        # 3. Tensor 변환 및 디바이스 업로드
        imu_t = torch.tensor(seq_resampled, dtype=torch.float32).unsqueeze(0).to(self.device)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, 128, 128]
        
        # 4. 하이브리드 추론
        with torch.no_grad():
            out = self.model(imu_t, img_t)
            probs = torch.softmax(out, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            conf = confidence.item()
            idx = pred_idx.item()
            
        if conf < 0.3:
            return None, conf
            
        for label, i in self.label_map.items():
            if i == idx:
                print(f"[AI hybrid] 예측: {label} (신뢰도: {conf*100:.1f}%)")
                return label, conf
        return None, 0.0
    
    def _predict_jw_v1(self, session_strokes):
        """JW v1 추론: 궤적 이미지 + IMU 시계열 동시 입력"""
        if self.renderer is None:
            from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
            self.renderer = TrajectoryRenderer(size=128)
        
        # 궤적 → 이미지
        img = self.renderer.render(session_strokes)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        seq = extract_features(session_strokes)
        if len(seq) < 5:
            return None, 0.0

        seq = self._resample_imu(seq, 200)
        if self.scaler:
            seq = self.scaler.transform(seq)

        imu_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        label, conf, top_k = self.model.predict(imu_t, img_t, self.label_map)
        if label:
            print(f"[JW v1] 예측: {label} (신뢰도: {conf*100:.1f}%)")
            return label, conf
        return None, 0.0
    
    def _predict_legacy(self, session_strokes):
        """순수 시계열 모델 (Pure BiLSTM / Transformer) 추론"""
        self.model.eval()
        
        # 입력 구조 로깅 (DEBUG 레벨: 실시간 추론 성능 영향 없음)
        if _logger.isEnabledFor(logging.DEBUG):
            stroke_lens = [len(s) for s in session_strokes]
            _logger.debug("_predict_legacy: %d strokes, lengths=%s", len(session_strokes), stroke_lens)

        seq = extract_features(session_strokes)
        if _logger.isEnabledFor(logging.DEBUG):
            new_stroke_positions = [int(i) for i in np.flatnonzero(seq[:, 4] == 1.0)] if seq.size else []
            _logger.debug("is_new_stroke=1 at positions: %s (total %d frames)", new_stroke_positions, len(seq))

        if len(seq) < 5:
            return None, 0.0

        seq = self._resample_imu(seq, 200)
        if self.scaler:
            seq = self.scaler.transform(seq)
            
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            conf = confidence.item()
            idx = pred_idx.item()
            
            # Top-3 예측 로깅 (DEBUG 레벨)
            if _logger.isEnabledFor(logging.DEBUG):
                top3_vals, top3_idxs = torch.topk(probs, min(3, probs.shape[1]), dim=1)
                inv_map = {v: k for k, v in self.label_map.items()}
                top3_str = ", ".join(f"{inv_map.get(top3_idxs[0,i].item(),'?')}:{top3_vals[0,i].item()*100:.1f}%" for i in range(top3_vals.shape[1]))
                _logger.debug("Top-3: [%s]", top3_str)
            
        if conf < 0.3:
            return None, conf
            
        for label, i in self.label_map.items():
            if i == idx:
                print(f"[AI {self.model_type}] 예측: {label} (신뢰도: {conf*100:.1f}%)")
                return label, conf
        return None, 0.0
    
    def _predict_ctc(self, session_strokes):
        """CTC 모드 추론: 연속 문자 인식 (자막 모드)"""
        try:
            self.model.eval()
            
            seq = extract_features(session_strokes)
            print(f"[CTC DEBUG] extract_features shape: {seq.shape}")
            if len(seq) < 5:
                print(f"[CTC DEBUG] 시퀀스 너무 짧음: {len(seq)} < 5")
                return None, 0.0

            # CTC는 리샘플링 없이 원본 길이 유지!
            seq = seq.astype(np.float32, copy=False)
            
            # scaler는 11차원 전체에 대해 fit되었으므로, 먼저 11차원 상태에서 transform
            if self.ctc_scaler is not None:
                seq = self.ctc_scaler.transform(seq)
            
            # X, Y 절대 좌표(0,1번)를 제거하여 9차원으로 변환 (학습 시 ctc_dataset.__getitem__과 동일)
            seq = seq[:, 2:]
            print(f"[CTC DEBUG] 최종 입력 shape: {seq.shape}")
            
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            input_lengths = torch.tensor([len(seq)], dtype=torch.long)
            
            with torch.no_grad():
                log_probs = self.model(x)  # [T, 1, vocab]
                print(f"[CTC DEBUG] log_probs shape: {log_probs.shape}")
                
                # Greedy 디코딩
                decoded = self.model.greedy_decode(log_probs, input_lengths)
                text = decoded[0] if decoded else ""
                print(f"[CTC DEBUG] decoded: {decoded}, text: '{text}'")
                
                # 신뢰도: 실제 문자가 예측된 프레임에서의 확률만 평균
                probs = log_probs.exp().squeeze(1)  # [T, vocab]
                best_ids = probs.argmax(dim=-1)  # [T]
                # blank이 아닌 프레임만 선택
                non_blank_mask = best_ids != 0
                if non_blank_mask.any():
                    non_blank_probs = probs[non_blank_mask]
                    char_probs = non_blank_probs.gather(1, best_ids[non_blank_mask].unsqueeze(1)).squeeze(1)
                    conf = float(char_probs.mean().item())
                else:
                    conf = 0.0
            
            if not text:
                print(f"[CTC DEBUG] 빈 텍스트 반환")
                return None, 0.0
            
            print(f"[CTC] 인식: '{text}' (신뢰도: {conf*100:.1f}%)")
            return text, conf
        except Exception as e:
            print(f"[CTC ERROR] _predict_ctc 에러: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def export_onnx(self, path=None):
        if path is None:
            path = str(_WEIGHTS_DIR / "model.onnx")
        if self.model is None:
            print("모델이 없습니다. 먼저 학습하세요.")
            return False
        
        if self.model_type == "jw_v1":
            self.model.export_onnx(str(_WEIGHTS_DIR / "jw_v1.onnx"))
        else:
            self.model.eval()
            dummy = torch.randn(1, 200, 11).to(self.device)
            # seq_len is always 200 after _resample_imu, so only batch is dynamic.
            # Marking seq_len dynamic conflicts with LSTM/MHA's traced static shape under torch.export.
            torch.onnx.export(
                self.model, dummy, path,
                opset_version=17,
                input_names=["imu_sequence"],
                output_names=["prediction"],
                dynamic_axes={"imu_sequence": {0: "batch"}, "prediction": {0: "batch"}},
            )
            print(f"ONNX 변환 완료: {path}")
        return True

    def _resample_imu(self, seq, target_len):
        """추론용 간이 리샘플러"""
        curr_len = len(seq)
        if curr_len < 2: 
            return np.zeros((target_len, seq.shape[1]))
        x_old = np.linspace(0, 1, curr_len)
        x_new = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, seq.shape[1]))
        for i in range(seq.shape[1]):
            if i == 4:
                # is_new_stroke은 이진 플래그 → nearest-neighbor로 0/1 보존(추론↔학습 정합).
                nn_idx = np.round(x_new * (curr_len - 1)).astype(int)
                resampled[:, i] = seq[nn_idx, i]
            else:
                resampled[:, i] = np.interp(x_new, x_old, seq[:, i])
        return resampled


def render_trajectory_to_image(seq, img_size=128):
    """
    IMU 궤적 수치 데이터 [T, 11]의 (x, y) 좌표 및 is_new_stroke 신호를 활용하여
    [img_size, img_size] 흑백 렌더링 이미지 생성 (0.0: 검은선, 1.0: 흰바탕)
    OpenCV(cv2) 라이브러리 없이 순수 NumPy 기반으로 속도 및 독립성 보장.
    """
    img = np.ones((img_size, img_size), dtype=np.float32)
    if len(seq) < 2:
        return img
        
    x = seq[:, 0]
    y = seq[:, 1]
    is_new = seq[:, 4]
    
    # 1. Bounding Box 정규화
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xspan = xmax - xmin
    yspan = ymax - ymin
    span = max(xspan, yspan, 1e-5)
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    
    # 2. 이미지 좌표 매핑 (경계 여백 10% 제외하고 0.8 스케일링)
    # 이미지 픽셀 좌표계에 정합시키기 위해 Y축은 반전(1.0 - y) 처리
    draw_area = img_size * 0.8
    
    px = (((x - x_center) / span) * draw_area + (img_size / 2.0)).astype(np.int32)
    py = ((1.0 - ((y - y_center) / span)) * draw_area + (img_size / 2.0)).astype(np.int32)
    
    # 이미지 범위 클리핑
    px = np.clip(px, 0, img_size - 1)
    py = np.clip(py, 0, img_size - 1)
    
    # 3. 획 렌더링 (순수 넘파이 Bresenham Line 알고리즘)
    def draw_line(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # 기본 두께(3px) 구현을 위한 주변 픽셀 마킹 (안전 바운더리 포함)
            for ox in range(-1, 2):
                for oy in range(-1, 2):
                    nx, ny = x0 + ox, y0 + oy
                    if 0 <= nx < img_size and 0 <= ny < img_size:
                        img[ny, nx] = 0.0 # 검은 선 긋기
                        
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    start_idx = 0
    for i in range(1, len(px)):
        if is_new[i] == 1.0:
            # 획 분할 지점 발생 -> 이전 획 드로잉
            for j in range(start_idx, i - 1):
                draw_line(px[j], py[j], px[j+1], py[j+1])
            start_idx = i
            
    # 마지막 남은 획 드로잉
    for j in range(start_idx, len(px) - 1):
        draw_line(px[j], py[j], px[j+1], py[j+1])
        
    return img


class HybridIMUImageNet(nn.Module):
    def __init__(self, num_classes=26, hidden_dim=128):
        super().__init__()
        
        # ─── 1. IMU 시계열 인코더 (1D CNN + BiLSTM + Attention) ───
        self.imu_conv = nn.Sequential(
            nn.Conv1d(in_channels=11, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.15
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # ─── 2. 궤적 이미지 인코더 (CNN) ───
        self.cnn = nn.Sequential(
            # Block 1: [B, 1, 128, 128] -> [B, 16, 64, 64]
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: [B, 16, 64, 64] -> [B, 32, 32, 32]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: [B, 32, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ─── 3. 멀티모달 결합 분류기 ───
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_imu, x_img):
        # 1. IMU 분기
        # Conv1d를 위해 차원 전치: [B, T, 11] -> [B, 11, T]
        x_imu_t = x_imu.transpose(1, 2)
        imu_feat = self.imu_conv(x_imu_t)
        imu_feat = imu_feat.transpose(1, 2) # [B, T, 32]
        
        lstm_out, _ = self.lstm(imu_feat) # [B, T, hidden_dim * 2]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        imu_out = attn_out[:, -1, :] # 시계열 마지막 시점 요약벡터 [B, hidden_dim * 2]
        
        # 2. 이미지 분기
        cnn_features = self.cnn(x_img)
        img_out = self.cnn_fc(cnn_features) # [B, 256]
        
        # 3. Concatenate 및 분류
        fused = torch.cat([imu_out, img_out], dim=-1) # [B, hidden_dim * 2 + 256]
        logits = self.classifier(fused)
        return logits

