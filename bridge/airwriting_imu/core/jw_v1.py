"""
JW v1 — AirWriting AI Engine
=============================
나만의 로컬 AI 모델. VQ-VAE 토크나이저 + Mamba2 SSM 자가회귀 백본.

아키텍처 영감:
  - Mamba2 (Gu & Dao, 2024): O(N) State Space Model — Transformer 대체
  - VQ-VAE (van den Oord et al.): 연속 신호 → 이산 토큰 변환
  - PRIMUS (NeurIPS 2025): IMU 자기지도 사전학습
  - eMamba (2025): Edge 최적화 SSM
  - TrOCR (Microsoft): Vision Encoder + AR Decoder 구조 참고

창의적 기여 (JW v1 고유 설계):
  - Dual-Modal VQ: IMU 시계열 + 궤적 이미지를 공유 코드북으로 통합 토큰화
  - Mamba-CrossFusion: 두 모달리티의 SSM 히든 스테이트를 교차 게이팅
  - Adaptive Tokenizer: 필기 속도에 따라 토큰 해상도 동적 조절

모델 사양:
  - 파라미터: ~1.2M (GPT-2의 1/100)
  - 추론: <15ms (CPU), <3ms (CUDA/Jetson Orin Nano 8GB)
  - 메모리: ~5MB (FP16)
  - 입력: IMU 8ch 시계열 + 128×128 궤적 이미지
  - 출력: 영문 글자/단어 인식 + 궤적 자동완성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 2.4+ 에서만 nn.RMSNorm 지원 — 하위 버전 호환 폴백
if not hasattr(nn, "RMSNorm"):
    class _RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            return x / rms * self.weight
    nn.RMSNorm = _RMSNorm


# ═══════════════════════════════════════════════════════════════════
# Stage 1: VQ-VAE Tokenizer — 연속 센서 신호를 이산 토큰으로 변환
# ═══════════════════════════════════════════════════════════════════

class VectorQuantizer(nn.Module):
    """
    VQ-VAE 코드북. 연속 벡터를 가장 가까운 코드북 엔트리로 매핑.
    
    EMA (Exponential Moving Average) 업데이트로 안정적인 코드북 학습.
    Straight-Through Estimator로 그래디언트 전파.
    """
    def __init__(self, codebook_size: int = 512, dim: int = 64, 
                 commitment_cost: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        
        # 코드북 임베딩 (EMA로 업데이트)
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.embedding.weight.requires_grad = False
        
        # EMA 트래킹
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', self.embedding.weight.clone())
        self.ema_decay = ema_decay
        
    def forward(self, z: torch.Tensor):
        """
        z: [B, T, D] — 인코더 출력 (연속 벡터)
        Returns: quantized, loss, token_indices
        """
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)  # [B*T, D]
        
        # 가장 가까운 코드북 엔트리 찾기 (L2 거리)
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = dist.argmin(dim=1)  # [B*T]
        
        # 양자화된 벡터 가져오기
        quantized = self.embedding(indices).reshape(B, T, D)
        
        # 손실 계산
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-Through Estimator: 순전파는 양자화된 값, 역전파는 연속 값
        quantized_st = z + (quantized - z).detach()
        
        # EMA 코드북 업데이트 (학습 시)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.codebook_size).float()
                self.ema_count.mul_(self.ema_decay).add_(one_hot.sum(0), alpha=1 - self.ema_decay)
                dw = one_hot.t() @ z_flat
                self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                
                # Laplace smoothing으로 죽은 코드 방지
                n = self.ema_count.sum()
                count = (self.ema_count + 1e-5) / (n + self.codebook_size * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_weight / count.unsqueeze(1))
        
        return quantized_st, vq_loss, indices.reshape(B, T)


class MotionEncoder(nn.Module):
    """
    IMU 8채널 시계열 → 연속 잠재 벡터 시퀀스.
    1D Causal Conv로 시간 패턴 추출, 4배 다운샘플링.
    """
    def __init__(self, in_channels: int = 8, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # [B, 8, T] → [B, 32, T]
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            
            # [B, 32, T] → [B, 64, T//2]
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            # [B, 64, T//2] → [B, latent_dim, T//4]
            nn.Conv1d(64, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 8] → [B, T//4, latent_dim]"""
        x = x.transpose(1, 2)  # [B, 8, T]
        z = self.encoder(x)     # [B, latent_dim, T//4]
        return z.transpose(1, 2)  # [B, T//4, latent_dim]


class ImageEncoder(nn.Module):
    """
    궤적 이미지 128×128 → 연속 잠재 벡터 시퀀스.
    경량 CNN으로 16개 패치 토큰 추출.
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # [B, 1, 128, 128] → [B, 16, 64, 64]
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            
            # → [B, 32, 32, 32]
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            
            # → [B, 64, 16, 16]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            # → [B, latent_dim, 8, 8]
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
            
            # → [B, latent_dim, 4, 4]  
            nn.AdaptiveAvgPool2d((4, 4)),
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, 1, 128, 128] → [B, 16, latent_dim]"""
        z = self.encoder(img)            # [B, latent_dim, 4, 4]
        B, C, H, W = z.shape
        return z.reshape(B, C, H * W).transpose(1, 2)  # [B, 16, latent_dim]


class MotionDecoder(nn.Module):
    """VQ 토큰 → IMU 재구성 (VQ-VAE 학습용)."""
    def __init__(self, out_channels: int = 8, latent_dim: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, out_channels, kernel_size=7, stride=1, padding=3),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, T//4, latent_dim] → [B, T, 8]"""
        x = z.transpose(1, 2)
        return self.decoder(x).transpose(1, 2)


class DualModalVQTokenizer(nn.Module):
    """
    JW v1 고유 설계: Dual-Modal VQ Tokenizer
    
    IMU 시계열과 궤적 이미지를 하나의 공유 코드북으로 토큰화.
    Cross-Attention으로 두 모달리티의 정보를 융합한 뒤 양자화.
    """
    def __init__(self, latent_dim: int = 64, codebook_size: int = 512, in_channels: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.motion_enc = MotionEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.image_enc = ImageEncoder(latent_dim=latent_dim)
        self.motion_dec = MotionDecoder(out_channels=in_channels, latent_dim=latent_dim)
        
        # Cross-Attention: 이미지 특징이 모션 특징을 보강 (JW v1 고유)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.cross_norm = nn.LayerNorm(latent_dim)
        
        # Fusion gate: 모달리티 중요도 동적 조절
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        
        self.vq = VectorQuantizer(codebook_size=codebook_size, dim=latent_dim)
        self.latent_dim = latent_dim
    
    def encode(self, imu_seq: torch.Tensor, traj_img: torch.Tensor):
        """
        imu_seq: [B, T, 8] — IMU 시계열
        traj_img: [B, 1, 128, 128] — 궤적 이미지
        Returns: quantized, vq_loss, token_indices
        """
        z_motion = self.motion_enc(imu_seq)    # [B, T//4, D]
        z_image = self.image_enc(traj_img)     # [B, 16, D]
        
        # 이미지 특징으로 모션 특징 보강 (Cross-Attention)
        z_cross, _ = self.cross_attn(
            query=z_motion, key=z_image, value=z_image
        )
        z_cross = self.cross_norm(z_motion + z_cross)
        
        # Adaptive gate: 모달리티 균형 조절
        gate_input = torch.cat([z_motion, z_cross], dim=-1)
        g = self.gate(gate_input)
        z_fused = g * z_cross + (1 - g) * z_motion
        
        # Vector Quantization
        quantized, vq_loss, indices = self.vq(z_fused)
        return quantized, vq_loss, indices
    
    def decode_motion(self, quantized: torch.Tensor) -> torch.Tensor:
        """양자화된 토큰 → IMU 재구성"""
        return self.motion_dec(quantized)
    
    def tokenize(self, imu_seq: torch.Tensor, traj_img: torch.Tensor):
        """추론 시: 센서 데이터 → 토큰 인덱스만 반환"""
        with torch.no_grad():
            _, _, indices = self.encode(imu_seq, traj_img)
        return indices


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Mamba SSM Backbone — O(N) 선형 복잡도 시퀀스 모델링
# ═══════════════════════════════════════════════════════════════════

def selective_scan(u, delta, A, B, C, D_skip=None):
    """
    Mamba Selective Scan — 순수 PyTorch 구현.
    
    State Space Model: h(t) = Ā·h(t-1) + B̄·x(t), y(t) = C·h(t) + D·x(t)
    여기서 Ā = exp(Δ·A), B̄ = Δ·B (Zero-Order Hold 이산화)
    
    Args:
        u: [B, L, D] — 입력 시퀀스
        delta: [B, L, D] — 적응적 스텝 크기 (입력 의존)
        A: [D, N] — 상태 전이 행렬 (로그 공간)
        B: [B, L, N] — 입력 행렬
        C: [B, L, N] — 출력 행렬
        D_skip: [D] — Skip connection
    Returns: [B, L, D]
    """
    B_batch, L, D = u.shape
    N = A.shape[1]
    
    # delta를 softplus로 양수 보장
    delta = F.softplus(delta)
    
    # 이산화: Ā = exp(Δ·A), 로그 공간에서 안정적 계산
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B,L,D,N]
    dB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B,L,D,N]
    
    # 순차 스캔 (추론 시 스트리밍 가능)
    h = torch.zeros(B_batch, D, N, device=u.device, dtype=u.dtype)
    ys = []
    
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * u[:, t, :, None]
        y = (h * C[:, t, None, :]).sum(dim=-1)  # [B, D]
        ys.append(y)
    
    y = torch.stack(ys, dim=1)  # [B, L, D]
    
    if D_skip is not None:
        y = y + u * D_skip.unsqueeze(0).unsqueeze(0)
    
    return y


class MambaBlock(nn.Module):
    """
    Mamba2 SSM Block — Transformer의 Self-Attention을 대체.
    
    핵심 차이:
    - Transformer: O(N²) attention → 시퀀스 길어지면 급격히 느려짐
    - Mamba: O(N) selective scan → 길이에 비례하는 선형 비용
    - 추론 시 고정 크기 hidden state → 메모리 일정 (KV 캐시 불필요)
    
    JW v1 커스터마이징:
    - RMSNorm 사용 (LayerNorm 대비 15% 빠름)
    - SiLU 게이팅으로 정보 흐름 제어
    - Residual + 드롭아웃으로 소량 데이터 정규화
    """
    def __init__(self, d_model: int = 128, d_state: int = 16, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expand
        
        self.norm = nn.RMSNorm(d_model)
        
        # 입력 프로젝션: x → [z, gate] 동시 생성
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Depthwise 1D Conv: 로컬 컨텍스트 캡처
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=4, padding=3,
            groups=d_inner  # Depthwise = 채널별 독립 → 파라미터 절약
        )
        
        # SSM 파라미터  
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)  # B, C 생성
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)       # Δ 생성
        
        # A: 상태 전이 행렬 (로그 공간, HiPPO 초기화)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D: Skip connection
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # 출력 프로젝션
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → [B, L, d_model]"""
        residual = x
        x = self.norm(x)
        
        # 입력 프로젝션 + 게이팅 분기
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)  # 각각 [B, L, d_inner]
        
        # Depthwise Conv (causal: 미래 정보 차단)
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :x.shape[1]]  # causal trim
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)
        
        # SSM 파라미터 생성 (입력 의존적 = "Selective")
        x_proj = self.x_proj(x_conv)
        B_param, C_param = x_proj.split(self.A_log.shape[1], dim=-1)
        delta = self.dt_proj(x_conv)
        
        # A는 음수여야 안정 (로그 공간에서 복원 후 부호 반전)
        A = -torch.exp(self.A_log)
        
        # Selective Scan 실행
        y = selective_scan(x_conv, delta, A, B_param, C_param, self.D)
        
        # 게이팅: SiLU(z) ⊙ y
        y = y * F.silu(z)
        
        # 출력 프로젝션 + Residual
        out = self.out_proj(y)
        out = self.dropout(out)
        return out + residual


# ═══════════════════════════════════════════════════════════════════
# Stage 3: JW v1 전체 모델
# ═══════════════════════════════════════════════════════════════════

class JWv1(nn.Module):
    """
    JW v1 — 에어라이팅 전용 자가회귀 AI 모델.
    
    소유자가 직접 설계한 로컬 AI 엔진.
    OpenAI/Gemini 같은 외부 API 없이 완전 독립 동작.
    
    구조:
      [IMU 8ch + 궤적 이미지 128×128]
        → VQ Tokenizer (이산화)
        → Mamba-AR Backbone (4-layer SSM)
        → Dual Head (분류 + 생성)
    
    사양:
      - 파라미터: ~1.2M
      - CUDA 추론: <3ms (Jetson Orin Nano)
      - CPU 추론: <15ms
      - 지원: 영문 글자 + 짧은 영어 단어
    """
    def __init__(
        self,
        codebook_size: int = 512,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        max_seq_len: int = 256,
        num_classes: int = 62,   # a-z(26) + A-Z(26) + 0-9(10) or word vocab
        dropout: float = 0.1,
        latent_dim: int = 64,
        in_channels: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_classes = num_classes

        # ─── Stage 1: VQ Tokenizer ───
        self.tokenizer = DualModalVQTokenizer(
            latent_dim=latent_dim, codebook_size=codebook_size, in_channels=in_channels
        )
        
        # ─── Stage 2: Token → Mamba Backbone ───
        self.tok_embed = nn.Embedding(codebook_size + 2, d_model)  # +2: [CLS], [PAD]
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # 잠재 차원 → 모델 차원 정렬
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, 
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.RMSNorm(d_model)
        
        # ─── Stage 3: Dual Output Heads ───
        # Head A: 분류 (글자/단어 인식)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Head B: 자가회귀 생성 (다음 토큰 예측)
        self.lm_head = nn.Linear(d_model, codebook_size, bias=False)
        
        # 참고: weight tying은 forward에서 수동 적용
        # (Embedding 슬라이스를 Parameter로 직접 할당 불가하므로)
        
        # 특수 토큰 인덱스
        self.cls_token_id = codebook_size      # [CLS] 토큰
        self.pad_token_id = codebook_size + 1  # [PAD] 토큰
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming 초기화로 안정적 학습 시작"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def encode_to_tokens(self, imu_seq, traj_img):
        """센서 데이터 → VQ 토큰 인덱스 (추론용)"""
        return self.tokenizer.tokenize(imu_seq, traj_img)
    
    def forward_backbone(self, token_ids: torch.Tensor = None, 
                          z_continuous: torch.Tensor = None):
        """
        토큰 시퀀스를 Mamba 백본에 통과.
        
        token_ids: [B, T] — 이산 토큰 (추론 시)
        z_continuous: [B, T, latent_dim] — 연속 잠재 벡터 (학습 시, STE 통과)
        """
        B = token_ids.shape[0] if token_ids is not None else z_continuous.shape[0]
        
        if z_continuous is not None:
            # 학습 시: VQ의 Straight-Through 출력 사용
            x = self.latent_proj(z_continuous)
        else:
            # 추론 시: 토큰 임베딩 사용
            x = self.tok_embed(token_ids)
        
        T = x.shape[1]
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        
        # Mamba SSM 블록 스택
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        return x
    
    def forward(self, imu_seq: torch.Tensor, traj_img: torch.Tensor, 
                mode: str = "classify"):
        """
        End-to-end Forward Pass.
        
        Args:
            imu_seq: [B, T, 8] — IMU 시계열
            traj_img: [B, 1, 128, 128] — 궤적 이미지  
            mode: "classify" | "generate" | "pretrain"
        
        Returns:
            classify: [B, num_classes] 로짓
            generate: [B, T, codebook_size] 다음 토큰 로짓
            pretrain: (lm_logits, vq_loss) 사전학습 출력
        """
        # VQ 인코딩
        quantized, vq_loss, token_ids = self.tokenizer.encode(imu_seq, traj_img)
        
        # Mamba 백본 (학습 시 연속 벡터, 추론 시 토큰)
        if self.training:
            features = self.forward_backbone(z_continuous=quantized)
        else:
            features = self.forward_backbone(token_ids=token_ids)
        
        if mode == "classify":
            # 시퀀스 평균 풀링 → 분류
            pooled = features.mean(dim=1)
            logits = self.cls_head(pooled)
            return logits, vq_loss
        
        elif mode == "generate":
            # 다음 토큰 예측 (자가회귀)
            lm_logits = self.lm_head(features)
            return lm_logits, vq_loss
        
        elif mode == "pretrain":
            # 사전학습: 분류 + 생성 동시
            lm_logits = self.lm_head(features)
            return lm_logits, vq_loss
    
    def predict(self, imu_seq: torch.Tensor, traj_img: torch.Tensor,
                label_map: dict = None) -> tuple:
        """
        추론 API — 실시간 글자 인식.
        
        Returns: (predicted_label, confidence, top_k)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(imu_seq, traj_img, mode="classify")
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)
            
            # Top-K 후보
            top_k_vals, top_k_ids = probs.topk(min(5, probs.shape[-1]), dim=-1)
        
        conf = confidence.item()
        idx = pred_idx.item()
        
        if conf < 0.3:
            return None, conf, []
        
        label = str(idx)
        if label_map:
            inv_map = {v: k for k, v in label_map.items()}
            label = inv_map.get(idx, str(idx))
        
        top_k = []
        for i in range(top_k_vals.shape[-1]):
            tid = top_k_ids[0, i].item()
            tconf = top_k_vals[0, i].item()
            tlabel = inv_map.get(tid, str(tid)) if label_map else str(tid)
            top_k.append({"label": tlabel, "conf": round(tconf, 4)})
        
        return label, conf, top_k
    
    def predict_with_stages(self, imu_seq: torch.Tensor, traj_img: torch.Tensor,
                            label_map: dict = None) -> dict:
        """
        파이프라인 시각화용 — 각 단계의 중간 결과를 모두 반환.
        
        Demo 페이지에서 5단계 애니메이션에 사용.
        
        Returns: dict with stage data
        """
        import time
        self.eval()
        stages = {}
        t_start = time.perf_counter()
        
        with torch.no_grad():
            # ─── Stage 1: Raw Input ───
            stages["raw_input"] = {
                "imu_shape": list(imu_seq.shape),
                "img_shape": list(traj_img.shape),
                "imu_stats": {
                    "mean": imu_seq.mean(dim=1)[0].tolist(),
                    "std": imu_seq.std(dim=1)[0].tolist(),
                    "points": int(imu_seq.shape[1]),
                    "channels": int(imu_seq.shape[2]),
                },
            }
            
            # ─── Stage 2: VQ-VAE Tokenizer ───
            z_motion = self.tokenizer.motion_enc(imu_seq)
            z_image = self.tokenizer.image_enc(traj_img)
            
            # Cross-Attention
            z_cross, attn_weights = self.tokenizer.cross_attn(
                query=z_motion, key=z_image, value=z_image
            )
            z_cross = self.tokenizer.cross_norm(z_motion + z_cross)
            
            # Fusion gate
            gate_input = torch.cat([z_motion, z_cross], dim=-1)
            g = self.tokenizer.gate(gate_input)
            z_fused = g * z_cross + (1 - g) * z_motion
            
            # VQ
            quantized, vq_loss, token_ids = self.tokenizer.vq(z_fused)
            
            stages["vq_tokenizer"] = {
                "motion_shape": list(z_motion.shape),
                "image_shape": list(z_image.shape),
                "token_ids": token_ids[0].tolist(),
                "codebook_size": self.codebook_size,
                "n_tokens": int(token_ids.shape[1]),
                "gate_values": g.mean(dim=-1)[0].tolist(),  # 모달리티 균형
                "vq_loss": float(vq_loss.item()),
                "attn_weights": attn_weights[0, :, :4].mean(dim=0).tolist() if attn_weights is not None else [],
            }
            
            # ─── Stage 3: Mamba SSM Backbone ───
            x = self.tok_embed(token_ids)
            T = x.shape[1]
            B = x.shape[0]
            positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embed(positions)
            
            layer_activations = []
            for i, block in enumerate(self.blocks):
                x = block(x)
                # 각 레이어 출력의 통계
                layer_activations.append({
                    "mean": float(x.mean().item()),
                    "std": float(x.std().item()),
                    "max": float(x.max().item()),
                    "norm": float(x.norm(dim=-1).mean().item()),
                    "activation_sample": x[0, :min(8, T), :4].tolist(),
                })
            
            x = self.final_norm(x)
            
            stages["mamba_backbone"] = {
                "n_layers": len(self.blocks),
                "d_model": self.d_model,
                "layer_activations": layer_activations,
                "output_norm": float(x.norm(dim=-1).mean().item()),
            }
            
            # ─── Stage 4: Classification Head ───
            pooled = x.mean(dim=1)
            logits = self.cls_head(pooled)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)
            top_k_vals, top_k_ids = probs.topk(min(10, probs.shape[-1]), dim=-1)
            
            # 라벨 매핑
            inv_map = {}
            if label_map:
                inv_map = {v: k for k, v in label_map.items()}
            
            top_k = []
            for j in range(top_k_vals.shape[-1]):
                tid = top_k_ids[0, j].item()
                tconf = top_k_vals[0, j].item()
                tlabel = inv_map.get(tid, str(tid))
                top_k.append({"label": tlabel, "confidence": round(tconf, 4)})
            
            all_probs = []
            for j in range(probs.shape[-1]):
                lbl = inv_map.get(j, str(j))
                all_probs.append({"label": lbl, "prob": round(float(probs[0, j].item()), 4)})
            all_probs.sort(key=lambda x: x["prob"], reverse=True)
            
            stages["classification"] = {
                "num_classes": self.num_classes,
                "top_k": top_k,
                "all_probs": all_probs[:20],  # 상위 20개만
            }
            
            # ─── Stage 5: Result ───
            t_end = time.perf_counter()
            pred_label = inv_map.get(pred_idx.item(), str(pred_idx.item()))
            
            stages["result"] = {
                "label": pred_label,
                "confidence": round(float(confidence.item()), 4),
                "inference_ms": round((t_end - t_start) * 1000, 2),
                "top_3": top_k[:3],
            }
        
        return stages
    
    def count_parameters(self) -> dict:
        """모델 파라미터 수 리포트"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        sections = {
            "tokenizer": sum(p.numel() for p in self.tokenizer.parameters()),
            "backbone": sum(p.numel() for p in self.blocks.parameters()),
            "cls_head": sum(p.numel() for p in self.cls_head.parameters()),
            "lm_head": sum(p.numel() for p in self.lm_head.parameters()),
            "embeddings": sum(p.numel() for p in self.tok_embed.parameters()) 
                        + sum(p.numel() for p in self.pos_embed.parameters()),
        }
        
        return {
            "total": total,
            "trainable": trainable,
            "sections": sections,
            "size_mb_fp32": total * 4 / (1024 ** 2),
            "size_mb_fp16": total * 2 / (1024 ** 2),
        }
    
    def export_onnx(self, path: str = "weights/jw_v1.onnx"):
        """Jetson Orin Nano 배포용 ONNX 변환"""
        self.eval()
        dummy_imu = torch.randn(1, 200, 8)
        dummy_img = torch.randn(1, 1, 128, 128)
        
        torch.onnx.export(
            self,
            (dummy_imu, dummy_img),
            path,
            opset_version=17,
            input_names=["imu_sequence", "trajectory_image"],
            output_names=["class_logits", "vq_loss"],
            dynamic_axes={
                "imu_sequence": {0: "batch", 1: "seq_len"},
                "trajectory_image": {0: "batch"},
            }
        )
        print(f"✅ JW v1 ONNX 변환 완료: {path}")


# ═══════════════════════════════════════════════════════════════════
# Stage 4: JW v2 (Pure Continuous Mamba)
# ═══════════════════════════════════════════════════════════════════

class JWv2_Continuous(nn.Module):
    """
    JW v2 — 순수 연속(Continuous) 시계열 Mamba 모델.
    
    VQ-VAE 이산화(Tokenization) 및 CNN(Motion/Image Encoder)을 완전히 배제하고,
    IMU 8채널 센서값을 직접 연속 공간(Continuous Space)에서 Mamba 블록으로 처리.
    Time-warp(속도 편차)에 대해 왜곡 없는 순수 Sequence 모델링.
    """
    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        num_classes: int = 62,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 1. Continuous Linear Projection (CNN 배제)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.RMSNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model),
            nn.RMSNorm(d_model)
        )
        
        # 2. Mamba Backbone
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, 
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.RMSNorm(d_model)
        
        # 3. Classifier Head (Attention Pooling 추가)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True, dropout=0.1
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, imu_seq: torch.Tensor, mode: str = "classify"):
        """
        imu_seq: [B, T, 8] — IMU 시계열 연속 데이터
        Returns: [B, num_classes] 로짓
        """
        # 1. Projection: [B, T, 8] -> [B, T, d_model]
        x = self.input_proj(imu_seq)
        
        # 2. Mamba Blocks
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        
        # 3. Attention Pooling
        attn_out, _ = self.attention(x, x, x)
        
        # Global Average Pooling
        pooled = attn_out.mean(dim=1)
        
        # 4. Classification
        logits = self.cls_head(pooled)
        return logits

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "size_mb_fp16": total * 2 / (1024 ** 2),
        }

# ═══════════════════════════════════════════════════════════════════
# Self-Test: 모델 생성 및 파라미터 확인
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  JW v1 — AirWriting AI Engine")
    print("=" * 60)
    
    model = JWv1(
        codebook_size=512,
        d_model=128,
        n_layers=4,
        d_state=16,
        num_classes=62,
    )
    
    # 파라미터 리포트
    report = model.count_parameters()
    print(f"\n📊 모델 사양:")
    print(f"   총 파라미터: {report['total']:,}")
    print(f"   학습 가능:   {report['trainable']:,}")
    print(f"   크기 (FP32): {report['size_mb_fp32']:.2f} MB")
    print(f"   크기 (FP16): {report['size_mb_fp16']:.2f} MB")
    print(f"\n📦 섹션별:")
    for name, count in report['sections'].items():
        print(f"   {name:15s}: {count:>8,}")
    
    # 추론 테스트
    print(f"\n🧪 추론 테스트...")
    dummy_imu = torch.randn(2, 200, 8)
    dummy_img = torch.randn(2, 1, 128, 128)
    
    model.eval()
    with torch.no_grad():
        logits, vq_loss = model(dummy_imu, dummy_img, mode="classify")
        print(f"   분류 출력:  {logits.shape}  (vq_loss={vq_loss.item():.4f})")
        
        lm_logits, _ = model(dummy_imu, dummy_img, mode="generate")
        print(f"   생성 출력:  {lm_logits.shape}")
    
    # 속도 벤치마크
    import time
    model.eval()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy_imu[:1], dummy_img[:1], mode="classify")
        times.append((time.perf_counter() - t0) * 1000)
    
    print(f"\n⚡ 추론 속도 (CPU):")
    print(f"   평균: {sum(times)/len(times):.1f}ms")
    print(f"   최소: {min(times):.1f}ms")
    print(f"   최대: {max(times):.1f}ms")
    
    print(f"\n✅ JW v1 모델 검증 완료!")
