"""
ECHWR -- Error-enhanced Contrastive Handwriting Recognition
=============================================================

대조 학습을 통한 특징 표현 고도화.

핵심: 학습 시에만 '보조 브랜치'를 사용하고 추론 시 폐기 -> 제로 추론 오버헤드.

구성:
  1. AuxiliaryTextBranch: Transformer로 GT 텍스트 임베딩 (학습 전용)
  2. InBatchContrastiveLoss: 배치 내 센서-텍스트 쌍 정렬 (InfoNCE)
  3. ErrorBasedContrastiveLoss: 합성 하드 네거티브로 미세 차이 구별력 강화
  4. ECHWRTrainer: 이중 대조 손실 통합 학습기

성능:
  - 작가 독립적(WI): CER -7.4%
  - 작가 의존적(WD): CER -10.4%
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Auxiliary Text Branch (Training-only, discarded at inference)
# =====================================================================

class AuxiliaryTextBranch(nn.Module):
    """
    보조 텍스트 브랜치 -- 학습 전용.
    
    Ground-truth 텍스트를 Transformer 인코더로 임베딩.
    센서 신호 브랜치와의 정렬을 유도.
    학습 완료 후 폐기 -> 제로 추론 오버헤드.
    """
    def __init__(self, vocab_size: int = 128, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        
        # 문자 임베딩
        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 프로젝션 헤드 (대조 학습 공간으로)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        text_ids: [B, L] 문자 인덱스 시퀀스
        Returns: [B, d_model] 텍스트 임베딩
        """
        B, L = text_ids.shape
        pos = torch.arange(L, device=text_ids.device).unsqueeze(0).expand(B, -1)
        
        x = self.char_embed(text_ids) + self.pos_embed(pos)
        x = self.encoder(x)
        
        # 평균 풀링
        pooled = x.mean(dim=1)
        return self.proj(pooled)
    
    @staticmethod
    def text_to_ids(text: str, max_len: int = 64) -> torch.Tensor:
        """문자열 -> 인덱스 텐서."""
        ids = [ord(c) % 128 for c in text[:max_len]]
        ids += [0] * (max_len - len(ids))  # 패딩
        return torch.tensor(ids, dtype=torch.long)


# =====================================================================
# Contrastive Losses
# =====================================================================

class InBatchContrastiveLoss(nn.Module):
    """
    인배치 대조 손실 -- 배치 내 센서-텍스트 쌍 정렬.
    
    InfoNCE 변형: 일치하는 쌍은 가까이, 불일치 쌍은 멀리.
    Temperature scaling으로 학습 안정성 제어.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, sensor_embeds: torch.Tensor,
                text_embeds: torch.Tensor) -> torch.Tensor:
        """
        sensor_embeds: [B, D] 센서 브랜치 출력
        text_embeds: [B, D] 텍스트 브랜치 출력
        Returns: scalar loss
        """
        # L2 정규화
        sensor_norm = F.normalize(sensor_embeds, dim=-1)
        text_norm = F.normalize(text_embeds, dim=-1)
        
        # 유사도 행렬
        logits = sensor_norm @ text_norm.T / self.temperature  # [B, B]
        
        # 대각선이 양성 쌍
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # 양방향 InfoNCE
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        
        return (loss_s2t + loss_t2s) / 2


class ErrorBasedContrastiveLoss(nn.Module):
    """
    오류 기반 대조 손실 -- 합성 하드 네거티브 생성.
    
    삭제/삽입/대체 오류를 주입한 '합성 하드 네거티브'와
    올바른 신호를 구분하도록 훈련.
    미세 동작 차이 구별력 극대화.
    """
    def __init__(self, temperature: float = 0.05,
                 error_types: list = None):
        super().__init__()
        self.temperature = temperature
        self.error_types = error_types or ['delete', 'insert', 'substitute']
    
    def generate_hard_negatives(self, text_ids: torch.Tensor,
                                  n_negatives: int = 3) -> torch.Tensor:
        """
        GT 텍스트에 인위적 오류를 주입하여 하드 네거티브 생성.
        
        text_ids: [B, L]
        Returns: [B * n_negatives, L]
        """
        B, L = text_ids.shape
        negatives = []
        
        for _ in range(n_negatives):
            neg = text_ids.clone()
            error_type = random.choice(self.error_types)
            
            for b in range(B):
                seq = neg[b]
                non_pad = (seq > 0).sum().item()
                if non_pad < 2:
                    continue
                
                pos = random.randint(0, max(0, non_pad - 1))
                
                if error_type == 'delete':
                    # 한 문자 삭제 (뒤로 밀기)
                    seq[pos:non_pad-1] = seq[pos+1:non_pad].clone()
                    seq[non_pad-1] = 0
                    
                elif error_type == 'insert':
                    # 랜덤 문자 삽입
                    if non_pad < L:
                        seq[pos+1:non_pad+1] = seq[pos:non_pad].clone()
                        seq[pos] = random.randint(32, 126)
                    
                elif error_type == 'substitute':
                    # 한 문자 치환
                    seq[pos] = random.randint(32, 126)
            
            negatives.append(neg)
        
        return torch.cat(negatives, dim=0)  # [B * n_neg, L]
    
    def forward(self, sensor_embeds: torch.Tensor,
                positive_text_embeds: torch.Tensor,
                negative_text_embeds: torch.Tensor) -> torch.Tensor:
        """
        sensor_embeds: [B, D]
        positive_text_embeds: [B, D]
        negative_text_embeds: [B * n_neg, D]
        """
        B = sensor_embeds.shape[0]
        sensor_norm = F.normalize(sensor_embeds, dim=-1)
        pos_norm = F.normalize(positive_text_embeds, dim=-1)
        neg_norm = F.normalize(negative_text_embeds, dim=-1)
        
        # 양성 점수
        pos_scores = (sensor_norm * pos_norm).sum(dim=-1) / self.temperature  # [B]
        
        # 음성 점수
        neg_scores = sensor_norm @ neg_norm.T / self.temperature  # [B, B*n_neg]
        
        # 양성 + 음성 결합
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, 1 + B*n_neg]
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)  # 0번이 양성
        
        return F.cross_entropy(logits, labels)


# =====================================================================
# ECHWR Trainer: Dual Contrastive Loss Integration
# =====================================================================

class ECHWRTrainer:
    """
    ECHWR 통합 학습기.
    
    센서 인코더 + 보조 텍스트 브랜치를 이중 대조 손실로 동시 학습.
    학습 완료 후 보조 브랜치 폐기 -> 제로 추론 오버헤드.
    
    LoRA와 결합하여 1.1M 파라미터만으로 도메인 적응.
    """
    def __init__(self, sensor_encoder: nn.Module, d_model: int = 128,
                 lr: float = 1e-3, temperature: float = 0.07,
                 lambda_inbatch: float = 1.0, lambda_error: float = 0.5):
        self.sensor_encoder = sensor_encoder
        self.d_model = d_model
        
        # 보조 브랜치 (학습 전용)
        self.text_branch = AuxiliaryTextBranch(
            vocab_size=128, d_model=d_model, nhead=4, num_layers=2)
        
        # 센서 프로젝션 (대조 공간으로)
        self.sensor_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 손실 함수
        self.inbatch_loss = InBatchContrastiveLoss(temperature=temperature)
        self.error_loss = ErrorBasedContrastiveLoss(temperature=temperature * 0.7)
        
        self.lambda_inbatch = lambda_inbatch
        self.lambda_error = lambda_error
        
        # 옵티마이저
        all_params = (
            list(self.sensor_encoder.parameters()) +
            list(self.text_branch.parameters()) +
            list(self.sensor_proj.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
    
    def train_step(self, sensor_input: torch.Tensor,
                   text_ids: torch.Tensor,
                   task_loss: torch.Tensor = None) -> dict:
        """
        단일 학습 스텝.
        
        sensor_input: [B, T, C] IMU 시퀀스
        text_ids: [B, L] GT 텍스트 인덱스
        task_loss: 기존 분류/인식 손실 (추가 결합)
        """
        self.sensor_encoder.train()
        self.text_branch.train()
        
        # 센서 인코딩
        sensor_features = self.sensor_encoder(sensor_input)  # [B, D]
        if sensor_features.dim() == 3:
            sensor_features = sensor_features.mean(dim=1)
        sensor_embeds = self.sensor_proj(sensor_features)
        
        # 텍스트 인코딩 (양성)
        text_embeds = self.text_branch(text_ids)
        
        # 인배치 대조 손실
        l_inbatch = self.inbatch_loss(sensor_embeds, text_embeds)
        
        # 오류 기반 대조 손실
        neg_ids = self.error_loss.generate_hard_negatives(text_ids, n_negatives=3)
        neg_embeds = self.text_branch(neg_ids.to(text_ids.device))
        l_error = self.error_loss(sensor_embeds, text_embeds, neg_embeds)
        
        # 총 손실
        total_loss = (self.lambda_inbatch * l_inbatch +
                      self.lambda_error * l_error)
        
        if task_loss is not None:
            total_loss = total_loss + task_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.sensor_encoder.parameters()) +
            list(self.text_branch.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'inbatch_loss': l_inbatch.item(),
            'error_loss': l_error.item(),
        }
    
    def discard_auxiliary(self):
        """
        학습 완료 후 보조 브랜치 폐기.
        -> 제로 추론 오버헤드 달성.
        """
        del self.text_branch
        del self.sensor_proj
        del self.optimizer
        self.text_branch = None
        self.sensor_proj = None
        self.optimizer = None
        
        # 센서 인코더만 유지
        self.sensor_encoder.eval()
        return self.sensor_encoder


# =====================================================================
# Self-Test
# =====================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    print("=" * 60)
    print("  ECHWR - Error-enhanced Contrastive Learning")
    print("=" * 60)
    
    # 센서 인코더 (간이)
    sensor_enc = nn.Sequential(
        nn.Linear(8, 64),
        nn.GELU(),
        nn.Linear(64, 128),
    )
    
    trainer = ECHWRTrainer(sensor_enc, d_model=128)
    
    # 학습 시뮬레이션
    for step in range(5):
        sensor_input = torch.randn(4, 200, 8)  # batch=4
        texts = ["Hello", "World", "Test!", "AirWr"]
        text_ids = torch.stack([
            AuxiliaryTextBranch.text_to_ids(t) for t in texts
        ])
        
        metrics = trainer.train_step(sensor_input, text_ids)
        print(f"  Step {step+1}: {metrics}")
    
    # 보조 브랜치 폐기
    encoder = trainer.discard_auxiliary()
    print(f"\nAuxiliary discarded. Encoder params: "
          f"{sum(p.numel() for p in encoder.parameters()):,}")
    
    # 추론 (보조 브랜치 없이)
    with torch.no_grad():
        out = encoder(torch.randn(1, 200, 8))
        print(f"Inference output: {out.shape}")
    
    print("\nDone!")
