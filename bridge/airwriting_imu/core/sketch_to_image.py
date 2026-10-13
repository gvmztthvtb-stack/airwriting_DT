"""
Sketch-to-Image 실시간 생성기.
ControlNet Scribble + LCM-LoRA 기반.

Jetson Orin Nano: SD1.5 + ControlNet + LCM = ~5GB VRAM, ~1.5초/장
PC (RTX 3060+): ~0.5초/장
"""
import torch
import base64
import io
import logging

log = logging.getLogger(__name__)


class SketchToImageGenerator:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.loaded = False

    def load(self):
        """모델 로딩 (최초 1회, ~30초)"""
        try:
            from diffusers import (
                StableDiffusionControlNetPipeline,
                ControlNetModel,
                LCMScheduler,
            )
        except ImportError:
            log.error("diffusers 패키지가 필요합니다: pip install diffusers transformers accelerate")
            return False

        log.info("🎨 Sketch-to-Image 모델 로딩 중... (최초 1회, ~30초)")

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_scribble",
            torch_dtype=torch.float16,
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)

        # LCM-LoRA 적용 (4 step으로 고속 생성)
        self.pipe.load_lora_weights(
            "latent-consistency/lcm-lora-sdv1-5"
        )
        self.pipe.fuse_lora()
        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config
        )

        # 메모리 최적화
        self.pipe.enable_attention_slicing()

        self.loaded = True
        log.info("🎨 Sketch-to-Image 모델 로딩 완료!")
        return True

    def generate(self, sketch_base64: str, prompt: str = "a flower") -> str:
        """스케치 이미지 → AI 생성 이미지 (base64 반환)"""
        if not self.loaded:
            if not self.load():
                return ""

        from PIL import Image

        # Base64 → PIL Image
        try:
            img_data = base64.b64decode(sketch_base64.split(",")[1])
        except (IndexError, Exception):
            img_data = base64.b64decode(sketch_base64)
        sketch = Image.open(io.BytesIO(img_data)).convert("RGB")
        sketch = sketch.resize((512, 512))

        # 생성
        result = self.pipe(
            prompt=f"high quality, detailed, {prompt}",
            negative_prompt="blurry, low quality, ugly",
            image=sketch,
            num_inference_steps=4,     # LCM: 4 step으로 충분
            guidance_scale=1.5,        # LCM은 낮은 CFG
            controlnet_conditioning_scale=0.8,
        ).images[0]

        # PIL → Base64
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(
            buf.getvalue()
        ).decode()
