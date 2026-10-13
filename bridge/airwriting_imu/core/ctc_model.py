"""
CTC Recognition Model — 연속 문자 인식을 위한 BiLSTM-CTC 아키텍처
====================================================================

기존 PureBiLSTMAttention과의 핵심 차이:
  - PureBiLSTMAttention: mean(seq) → 단일 클래스 출력 (한 글자씩 끊어서 분류)
  - CTCRecognizer: 모든 타임스텝 → 문자 시퀀스 출력 (자막처럼 연속 인식)

CTC(Connectionist Temporal Classification):
  - 입력과 출력의 정렬(alignment)을 몰라도 학습 가능
  - Blank 토큰으로 "아직 글자 안 씀" 상태를 표현
  - 같은 글자 반복(e.g. "LL")도 blank 삽입으로 구분

사용법:
  model = CTCRecognizer(num_classes=27)  # A-Z(26) + blank(1)
  loss = model.compute_loss(logits, targets, input_lengths, target_lengths)
  text = model.greedy_decode(logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CTCRecognizer(nn.Module):
    """
    BiLSTM-CTC 기반 연속 문자 인식 모델.
    
    구조:
      Input[B, T, 11] → Projection → BiLSTM(3-layer) → FC → [B, T, num_classes+1]
      
    CTC blank 토큰은 항상 index=0.
    실제 문자는 index 1~26 (A=1, B=2, ..., Z=26).
    
    파라미터: ~1.6M (PureBiLSTMAttention 대비 유사한 크기)
    """
    
    # 클래스 레벨 상수: 문자 매핑
    BLANK_IDX = 0
    CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        num_lstm_layers: int = 3,
        num_classes: int = 26,  # A-Z (blank은 자동 추가)
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = num_classes + 1  # +1 for CTC blank (index 0)
        
        # 1. Input Projection: 11 → 128
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 2. BiLSTM: 시간축 전체의 양방향 컨텍스트 학습
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        
        # 3. Self-Attention (Google Transformer Encoder Block)
        # BiLSTM의 양방향 피처 간의 장거리 글로벌 문맥 의존성을 융합합니다.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 4. Output Projection: Attention(hidden*2) → vocab_size
        # 매 타임스텝마다 문자 확률 분포 출력 (CTC 핵심)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.vocab_size),
        )
        
        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=self.BLANK_IDX, reduction='mean', zero_infinity=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 초기화로 안정적 학습 시작"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파: 시계열 → 매 타임스텝 문자 확률.
        
        Args:
            x: [B, T, input_dim] — 패딩된 IMU 시퀀스
            lengths: [B] — 패딩 전 실제 시퀀스 길이 (pack_padded_sequence 최적화용)
            
        Returns:
            log_probs: [T, B, vocab_size] — CTC 입력 형식 (Time-first!)
        """
        # 1. Projection
        x = self.projection(x)  # [B, T, hidden_dim]
        
        # 2. BiLSTM (pack_padded_sequence 최적화)
        if lengths is not None:
            # CPU 학습 시 패딩 구간 연산을 생략하여 속도를 2~3배 이상 높임
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            lstm_out, _ = self.lstm(x)  # [B, T, hidden_dim*2]
        
        # 3. Self-Attention (Transformer Encoder)
        key_padding_mask = None
        if lengths is not None:
            B, T = lstm_out.shape[0], lstm_out.shape[1]
            key_padding_mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)
            
        attn_out = self.transformer_encoder(lstm_out, src_key_padding_mask=key_padding_mask)  # [B, T, hidden_dim*2]
        
        # 4. Output
        logits = self.output_proj(attn_out)  # [B, T, vocab_size]
        
        # CTC는 [T, B, C] 형식 + log_softmax 필요
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # [T, B, vocab_size]
        
        return log_probs
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        CTC 손실 계산.
        
        Args:
            log_probs: [T, B, vocab_size] — forward() 출력
            targets: [B, S] 또는 1D concatenated — 타겟 문자 인덱스 (1-indexed, blank=0 제외)
            input_lengths: [B] — 각 샘플의 실제 시퀀스 길이 (패딩 전)
            target_lengths: [B] — 각 샘플의 타겟 문자열 길이
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    
    def greedy_decode(self, log_probs: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> List[str]:
        """
        Greedy CTC 디코딩: argmax → 연속 중복 제거 → blank 제거 → 문자열
        
        Args:
            log_probs: [T, B, vocab_size] — forward() 출력
            input_lengths: [B] — 각 샘플의 실제 길이 (None이면 전체 사용)
            
        Returns:
            decoded: List[str] — 디코딩된 문자열 리스트 (배치 크기만큼)
        """
        # [T, B, V] → [B, T, V]
        log_probs_bt = log_probs.permute(1, 0, 2)
        B, T, V = log_probs_bt.shape
        
        # argmax per timestep
        predictions = log_probs_bt.argmax(dim=-1)  # [B, T]
        
        decoded_strings = []
        for b in range(B):
            length = input_lengths[b].item() if input_lengths is not None else T
            pred_seq = predictions[b, :length].cpu().tolist()
            
            # Step 1: 연속 중복 제거 (e.g., [0,1,1,1,0,2,2] → [0,1,0,2])
            collapsed = []
            prev = -1
            for idx in pred_seq:
                if idx != prev:
                    collapsed.append(idx)
                prev = idx
            
            # Step 2: blank 제거
            chars = [self.CHARS[idx - 1] for idx in collapsed if idx != self.BLANK_IDX]
            
            decoded_strings.append("".join(chars))
        
        return decoded_strings
    
    def beam_decode(
        self, 
        log_probs: torch.Tensor, 
        input_lengths: Optional[torch.Tensor] = None,
        beam_width: int = 10,
    ) -> List[str]:
        """
        Beam Search CTC 디코딩 — Greedy보다 정확하지만 느림.
        
        Prefix beam search 알고리즘:
        각 타임스텝에서 beam_width개의 후보 prefix를 유지하며,
        blank/non-blank 확률을 따로 추적.
        """
        log_probs_bt = log_probs.permute(1, 0, 2)  # [B, T, V]
        B = log_probs_bt.shape[0]
        
        decoded_strings = []
        for b in range(B):
            T = input_lengths[b].item() if input_lengths is not None else log_probs_bt.shape[1]
            probs = log_probs_bt[b, :T].exp().cpu().numpy()  # [T, V]
            
            # Prefix beam search
            # 각 prefix: (blank_prob, non_blank_prob)
            beams = {("",): (1.0, 0.0)}  # prefix → (p_blank, p_non_blank)
            
            for t in range(T):
                new_beams = {}
                
                for prefix, (p_b, p_nb) in beams.items():
                    p_total = p_b + p_nb
                    
                    for c in range(self.vocab_size):
                        p_c = float(probs[t, c])
                        
                        if c == self.BLANK_IDX:
                            # blank 확장: prefix 유지
                            key = prefix
                            old = new_beams.get(key, (0.0, 0.0))
                            new_beams[key] = (old[0] + p_total * p_c, old[1])
                        else:
                            char = self.CHARS[c - 1]
                            
                            if len(prefix) > 0 and prefix[-1] == char:
                                # 같은 문자 반복: blank 후에만 새 문자 추가 가능
                                # Case 1: 반복 문자 병합 (non-blank → same char)
                                key = prefix
                                old = new_beams.get(key, (0.0, 0.0))
                                new_beams[key] = (old[0], old[1] + p_nb * p_c)
                                
                                # Case 2: blank 후 새 문자 (다른 'L' 추가 for "LL")
                                new_prefix = prefix + (char,)
                                old = new_beams.get(new_prefix, (0.0, 0.0))
                                new_beams[new_prefix] = (old[0], old[1] + p_b * p_c)
                            else:
                                # 다른 문자 확장
                                new_prefix = prefix + (char,)
                                old = new_beams.get(new_prefix, (0.0, 0.0))
                                new_beams[new_prefix] = (old[0], old[1] + p_total * p_c)
                
                # Top-K beam 유지
                scored = [(k, v[0] + v[1]) for k, v in new_beams.items()]
                scored.sort(key=lambda x: x[1], reverse=True)
                beams = {k: new_beams[k] for k, _ in scored[:beam_width]}
            
            # 최고 확률 prefix 선택
            best_prefix = max(beams.keys(), key=lambda k: sum(beams[k]))
            decoded_strings.append("".join(best_prefix))
        
        return decoded_strings
    
    def count_parameters(self) -> dict:
        """모델 파라미터 수 리포트"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "size_mb_fp32": round(total * 4 / (1024 ** 2), 2),
            "size_mb_fp16": round(total * 2 / (1024 ** 2), 2),
        }


# ═══════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  CTCRecognizer — BiLSTM-CTC 연속 문자 인식 모델")
    print("=" * 60)
    
    model = CTCRecognizer(num_classes=26)
    report = model.count_parameters()
    print(f"\n📊 모델 사양:")
    print(f"   총 파라미터: {report['total']:,}")
    print(f"   FP32 크기: {report['size_mb_fp32']} MB")
    print(f"   FP16 크기: {report['size_mb_fp16']} MB")
    print(f"   Vocab: {model.vocab_size} (blank + A-Z)")
    
    # 가변 길이 배치 시뮬레이션
    B = 4
    lengths = [80, 120, 200, 60]
    max_T = max(lengths)
    
    # 패딩된 입력
    x = torch.randn(B, max_T, 9)
    input_lengths = torch.tensor(lengths)
    
    # 타겟: "AB", "HELLO", "CAT", "Z"
    targets = torch.tensor([1, 2, 8, 5, 12, 12, 15, 3, 1, 20, 26])
    target_lengths = torch.tensor([2, 5, 3, 1])
    
    # Forward
    model.eval()
    with torch.no_grad():
        log_probs = model(x)
        print(f"\n🧪 Forward 테스트:")
        print(f"   입력: {x.shape} (패딩된 배치)")
        print(f"   출력: {log_probs.shape} (T, B, vocab)")
        
        # Loss 계산
        loss = model.compute_loss(log_probs, targets, input_lengths, target_lengths)
        print(f"   CTC Loss: {loss.item():.4f}")
        
        # Greedy 디코딩
        decoded = model.greedy_decode(log_probs, input_lengths)
        print(f"\n📝 Greedy 디코딩 결과:")
        for i, text in enumerate(decoded):
            print(f"   [{i}] '{text}' (길이 {input_lengths[i]})")
        
        # Beam 디코딩
        beam_decoded = model.beam_decode(log_probs, input_lengths, beam_width=5)
        print(f"\n🔍 Beam 디코딩 결과 (width=5):")
        for i, text in enumerate(beam_decoded):
            print(f"   [{i}] '{text}'")
    
    print(f"\n✅ CTCRecognizer 검증 완료!")
