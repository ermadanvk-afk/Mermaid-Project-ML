import torch
import gc
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler

class MemoryOrchestrator:
    """
    Manages loading and unloading of models between CPU RAM and GPU VRAM 
    to prevent Out-Of-Memory errors on constrained hardware.
    
    Supports LLaVA-NeXT-13B (4-bit NF4 quantized) and InstructPix2Pix (LCM).
    """
    def __init__(self, mllm_model_id="llava-hf/llava-v1.6-vicuna-13b-hf", diff_model_id="timbrooks/instruct-pix2pix"):
        self.mllm_loaded: bool = False
        self.diffusion_loaded: bool = False
        
        self.mllm_model_id = mllm_model_id
        self.diff_model_id = diff_model_id
        
        self.mllm = None
        self.processor = None
        self.diffusion_pipeline = None
        
        print(f"[MemoryManager] Initialized with MLLM: {self.mllm_model_id}")
        
    def _init_mllm(self):
        if self.mllm is None:
            print(f"[MemoryManager] Initializing MLLM ({self.mllm_model_id}) in 4-bit NF4...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.processor = LlavaNextProcessor.from_pretrained(self.mllm_model_id)
            self.mllm = LlavaNextForConditionalGeneration.from_pretrained(
                self.mllm_model_id,
                quantization_config=bnb_config,
                device_map={"": 0} if torch.cuda.is_available() else "auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
    def _init_diffusion(self):
        if self.diffusion_pipeline is None:
            print(f"[MemoryManager] Initializing Diffusion ({self.diff_model_id}) with LCM...")
            num_gpus = torch.cuda.device_count()
            
            if num_gpus > 1:
                # Kaggle T4 x2 Dual-GPU mode: dedicated GPU 1
                print("[MemoryManager] Dual-GPU detected: Loading Diffusion on cuda:1")
                self.diffusion_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    self.diff_model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to("cuda:1")
            else:
                # Single-GPU mode: enable CPU offloading to avoid VRAM clash with 13B model
                print("[MemoryManager] Single-GPU detected: Initializing Diffusion with CPU offload")
                self.diffusion_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    self.diff_model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None
                )
                if torch.cuda.is_available():
                    self.diffusion_pipeline.enable_model_cpu_offload(device="cuda:0")

            # Replace scheduler with LCM for fast 4-step generation
            try:
                self.diffusion_pipeline.scheduler = LCMScheduler.from_config(self.diffusion_pipeline.scheduler.config)
                self.diffusion_pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            except Exception as e:
                print(f"[MemoryManager] LCM LoRA not applied (continuing with standard steps): {e}")

    def load_mllm(self) -> None:
        """Loads MLLM context, clearing CUDA cache."""
        if self.mllm is None:
            self._init_mllm()
        torch.cuda.empty_cache()
        gc.collect()
        self.mllm_loaded = True

    def load_diffusion(self) -> None:
        """Prepares diffusion pipeline."""
        if self.diffusion_pipeline is None:
            self._init_diffusion()
        torch.cuda.empty_cache()
        gc.collect()
        self.diffusion_loaded = True

    def _offload_mllm(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        self.mllm_loaded = False

    def _offload_diffusion(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        self.diffusion_loaded = False
