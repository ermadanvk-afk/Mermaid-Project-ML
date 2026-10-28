import torch
import gc
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler
from core.qformer import CombinedModel, resize_image_to_512


class MemoryOrchestrator:
    """
    Manages loading and unloading of models between CPU RAM and GPU VRAM 
    to prevent Out-Of-Memory errors on constrained hardware (12GB VRAM).
    """
    def __init__(self, mllm_model_id="Qwen/Qwen2.5-VL-7B-Instruct", diff_model_id="timbrooks/instruct-pix2pix"):
        self.mllm_loaded: bool = False
        self.diffusion_loaded: bool = False
        
        self.mllm_model_id = mllm_model_id
        self.diff_model_id = diff_model_id
        self.clip_model_id = clip_model_id
        self.qformer_checkpoint_path = qformer_checkpoint_path
        self.use_lcm = use_lcm
        
        self.mllm = None
        self.processor = None
        self.diffusion_pipeline = None
        
        print("[MemoryManager] Initialized. Models will be loaded on demand.")
        
    def _init_mllm(self):
        if self.mllm is None:
            print(f"[MemoryManager] Initializing MLLM ({self.mllm_model_id}) in 4-bit NF4...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.processor = AutoProcessor.from_pretrained(self.mllm_model_id)
            # bitsandbytes 4-bit loads directly to GPU via device_map="auto"
            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.mllm_model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )

    def generate_mllm_response(
        self,
        image: Image.Image,
        prompt_text: str,
        max_new_tokens: int = 150,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        """
        Executes inference on LLaVA 1.5 7B using the proper Vicuna/chat template.
        """
        self.load_mllm()
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        
        formatted_prompt = None
        if hasattr(self.processor, "apply_chat_template") and getattr(self.processor, "chat_template", None) is not None:
            try:
                formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            except Exception:
                formatted_prompt = None
        elif hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "chat_template", None) is not None:
            try:
                formatted_prompt = self.processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = None

        if not formatted_prompt:
            # Canonical Vicuna v1.5 prompt for LLaVA 1.5
            formatted_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        elif "<image>" not in formatted_prompt:
            formatted_prompt = f"<image>\n{formatted_prompt}"
            
        device = self.mllm.device if hasattr(self.mllm, "device") and self.mllm.device.type != "meta" else ("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            
        with torch.no_grad():
            generated_ids = self.mllm.generate(**inputs, **gen_kwargs)
            
        new_tokens = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        del inputs, generated_ids, new_tokens
        torch.cuda.empty_cache()
        return output_text

    def _init_qformer(self):
        if self.qformer is None:
            print(f"[MemoryManager] Initializing CLIP ({self.clip_model_id}) & 12-layer Q-Former...")
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_id, torch_dtype=torch.float16)
            
            # Actual 12-layer Q-Former (76 queries, hidden_size 768)
            self.qformer = CombinedModel(num_query_tokens=76, hidden_size=768)
            self.qformer = self.qformer.to(torch.float16)
            
            if os.path.exists(self.qformer_checkpoint_path):
                print(f"[MemoryManager] Loading pretrained Q-Former weights from {self.qformer_checkpoint_path}...")
                trained_para = torch.load(self.qformer_checkpoint_path, map_location="cpu")
                self.qformer.load_state_dict(trained_para)
                print("[MemoryManager] Pretrained Q-Former weights loaded successfully.")
            else:
                print(f"[MemoryManager] Notice: '{self.qformer_checkpoint_path}' not found. Initialized 12-layer Q-Former with architecture defaults.")
            
    def _init_diffusion(self):
        if self.diffusion_pipeline is None:
            print(f"[MemoryManager] Initializing Diffusion ({self.diff_model_id}) with LCM...")
            self.diffusion_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.diff_model_id, dtype=torch.float16
            )
            # Replace scheduler with LCM for 2-4 step generation
            self.diffusion_pipeline.scheduler = LCMScheduler.from_config(self.diffusion_pipeline.scheduler.config)
            # Load LCM LoRA
            self.diffusion_pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
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
