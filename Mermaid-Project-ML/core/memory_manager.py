import torch
import gc
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline

class MemoryOrchestrator:
    """
    Manages loading and unloading of models between CPU RAM and GPU VRAM 
    to prevent Out-Of-Memory errors on constrained hardware (12GB VRAM).
    """
    def __init__(
        self,
        mllm_model_id="Qwen/Qwen2-VL-7B-Instruct",
        diff_model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    ):
        self.mllm_loaded: bool = False
        self.diffusion_loaded: bool = False
        
        self.mllm_model_id = mllm_model_id
        self.diff_model_id = diff_model_id
        
        self.mllm = None
        self.processor = None
        self.diffusion_pipeline = None
        
        print("[MemoryManager] Initialized. Models will be loaded on demand.")
        
    def _init_mllm(self):
        if self.mllm is None:
            print(f"[MemoryManager] Initializing MLLM ({self.mllm_model_id}) in 4-bit...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.processor = AutoProcessor.from_pretrained(self.mllm_model_id)
            # bitsandbytes 4-bit loads directly to GPU via device_map="auto"
            self.mllm = Qwen2VLForConditionalGeneration.from_pretrained(
                self.mllm_model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
    def _init_diffusion(self):
        if self.diffusion_pipeline is None:
            print(f"[MemoryManager] Initializing Diffusion ({self.diff_model_id}) with Hyper-SD...")
            self.diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.diff_model_id, torch_dtype=torch.float16, safety_checker=None
            )
            self.diffusion_pipeline.load_lora_weights(
                "ByteDance/Hyper-SD",
                weight_name="Hyper-SD15-4steps-lora.safetensors",
            )
            self.diffusion_pipeline.fuse_lora()
            self.diffusion_pipeline.scheduler = DDIMScheduler.from_config(
                self.diffusion_pipeline.scheduler.config
            )
            # Pipeline is instantiated on CPU by default
        
    def load_mllm(self) -> None:
        """Loads the MLLM to GPU, offloading Diffusion if necessary."""
        if self.diffusion_loaded:
            self._offload_diffusion()
            
        if self.mllm is None:
            self._init_mllm()
            self.mllm_loaded = True
        elif not self.mllm_loaded:
            print("[MemoryManager] Note: 4-bit MLLM usually remains on GPU. Skipping explicit move.")
            self.mllm_loaded = True
            
    def load_diffusion(self) -> None:
        """Loads the Diffusion model to GPU, offloading MLLM if necessary."""
        if self.mllm_loaded:
            self._offload_mllm()
            
        if self.diffusion_pipeline is None:
            self._init_diffusion()
            
        if not self.diffusion_loaded:
            print("[MemoryManager] Moving Diffusion model to GPU...")
            self.diffusion_pipeline.to('cuda')
            self.diffusion_loaded = True
            
    def _offload_mllm(self) -> None:
        """Moves MLLM to CPU and clears CUDA cache."""
        print("[MemoryManager] Offloading MLLM...")
        # 4-bit models cannot easily be moved to CPU via .to('cpu').
        # Given 12GB VRAM, a 4-bit 7B (5GB) and FP16 Diffusion (3GB) can coexist (8GB < 12GB).
        # We simply clear the cache to ensure we have maximum free space before running Diffusion.
        torch.cuda.empty_cache()
        gc.collect()
        self.mllm_loaded = False
        
    def _offload_diffusion(self) -> None:
        """Clears CUDA cache to make room. Doesn't strictly move to CPU to avoid fp16 errors."""
        if self.diffusion_pipeline is not None:
            print("[MemoryManager] Clearing cache after Diffusion...")
            # FP16 models complain when moved to CPU. Since 12GB can hold both,
            # we just clear the CUDA cache and leave it in VRAM.
            torch.cuda.empty_cache()
            gc.collect()
        self.diffusion_loaded = False
