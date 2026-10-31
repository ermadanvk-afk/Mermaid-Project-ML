import os
import torch
import gc
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
)
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler
from core.qformer import CombinedModel, resize_image_to_512


class MemoryOrchestrator:
    """
    Manages loading and unloading of models between CPU RAM and GPU VRAM
    to prevent Out-Of-Memory errors on constrained hardware (12 GB VRAM).

    MLLM  : LLaVA 1.5 7B (llava-hf/llava-1.5-7b-hf), loaded in
            4-bit NF4 quantisation via bitsandbytes for multimodal reasoning.
    Vision: 12-layer Q-Former + CLIP + InstructPix2Pix (with LCM LoRA)
            for emotion-conditioned visual augmentation.
    """
    def __init__(
        self,
        mllm_model_id: str = "llava-hf/llava-1.5-7b-hf",
        diff_model_id: str = "timbrooks/instruct-pix2pix",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        qformer_checkpoint_path: str = "checkpoint/Q-Former.pth",
        use_lcm: bool = True,
    ):
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
        
        # Q-Former & CLIP components for emotion augmentation
        self.qformer = None
        self.clip_model = None
        self.clip_processor = None
        
        print(f"[MemoryManager] Initialized with MLLM={mllm_model_id} & Q-Former. Models will be loaded on demand.")
        
    def _init_mllm(self):
        if self.mllm is None:
            print(f"[MemoryManager] Initializing MLLM ({self.mllm_model_id}) in 4-bit...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            is_next = "next" in self.mllm_model_id.lower() or "v1.6" in self.mllm_model_id.lower()
            if is_next:
                model_cls = LlavaNextForConditionalGeneration
                proc_cls = LlavaNextProcessor
            else:
                model_cls = LlavaForConditionalGeneration
                proc_cls = LlavaProcessor

            try:
                self.processor = proc_cls.from_pretrained(self.mllm_model_id)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(self.mllm_model_id)

            # bitsandbytes 4-bit loads directly to GPU via device_map="auto"
            self.mllm = model_cls.from_pretrained(
                self.mllm_model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
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
            print(f"[MemoryManager] Initializing Diffusion ({self.diff_model_id})...")
            self.diffusion_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.diff_model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )
            if self.use_lcm:
                print("[MemoryManager] Integrating LCM Scheduler and LoRA for accelerated augmentation...")
                self.diffusion_pipeline.scheduler = LCMScheduler.from_config(self.diffusion_pipeline.scheduler.config)
                try:
                    self.diffusion_pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                except (ValueError, OSError):
                    self.diffusion_pipeline.load_lora_weights(
                        "latent-consistency/lcm-lora-sdv1-5",
                        weight_name="pytorch_lora_weights.safetensors",
                    )
                
            # Enable attention slicing for minimal VRAM footprint
            self.diffusion_pipeline.enable_attention_slicing()
            self.diffusion_pipeline.set_progress_bar_config(disable=True)
        
    def load_mllm(self) -> None:
        """Ensures MLLM is loaded on GPU and clears transient cache."""
        if self.mllm is None:
            self._init_mllm()
            self.mllm_loaded = True
        torch.cuda.empty_cache()

    def load_diffusion(self) -> None:
        """Ensures Diffusion model, CLIP, and Q-Former are loaded on GPU."""
        if self.diffusion_pipeline is None:
            self._init_diffusion()
        if self.qformer is None:
            self._init_qformer()

        if not self.diffusion_loaded:
            print("[MemoryManager] Moving Diffusion pipeline, CLIP, and Q-Former to GPU...")
            self.diffusion_pipeline.to('cuda')
            self.clip_model.to('cuda')
            self.qformer.to('cuda')
            self.qformer.eval()
            self.clip_model.eval()
            self.diffusion_loaded = True
        torch.cuda.empty_cache()

    def _offload_mllm(self) -> None:
        """Cleans transient CUDA cache."""
        torch.cuda.empty_cache()
        gc.collect()

    def _offload_diffusion(self) -> None:
        """Cleans transient CUDA cache after diffusion inference."""
        torch.cuda.empty_cache()
        gc.collect()

    def extract_clip_image_embeds(self, image: Image.Image) -> torch.Tensor:
        """
        Extracts spatial patch CLIP image features ONCE per input image (resized to 512 short-side per A5)
        in torch.float16 on GPU for rich multi-emotion Q-Former cross-attention.
        Returns tensor of shape [1, 257, 768] (ViT-L patch tokens projected to Q-Former hidden size).
        """
        if not self.diffusion_loaded:
            self.load_diffusion()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        resized_img = resize_image_to_512(image)
        data = self.clip_processor(images=resized_img, return_tensors="pt")
        with torch.no_grad():
            pixel_vals = data["pixel_values"].to(device, dtype=torch.float16)
            v_out = self.clip_model.vision_model(pixel_values=pixel_vals)
            # Spatial patch tokens from ViT-L [1, 257, 1024]
            patch_tokens = v_out.last_hidden_state
            # Project to Q-Former hidden dimension (768)
            patch_proj = self.clip_model.visual_projection(patch_tokens)
            # L2 normalize per token
            patch_proj = patch_proj / patch_proj.norm(p=2, dim=-1, keepdim=True)
        return patch_proj.to(device, dtype=torch.float16)

    def get_emotion_prompt_embeds(
        self,
        image_or_emotion=None,
        emotion_or_none=None,
        img_embeds: torch.Tensor = None,
        emotion: str = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Fuses CLIP image features and target emotion embedding via the 12-layer Q-Former.
        Accepts:
          - (image, emotion, img_embeds=...) OR
          - (emotion, img_embeds=...) OR
          - (emotion=emotion, img_embeds=...)
        Produces prompt embeddings of shape [1, 77, 768] in float16 on GPU.
        """
        if not self.diffusion_loaded:
            self.load_diffusion()

        # Resolve emotion and image flexibly
        target_emotion = emotion or kwargs.get("emotion")
        if target_emotion is None:
            if emotion_or_none is not None:
                image = image_or_emotion
                target_emotion = emotion_or_none
            elif isinstance(image_or_emotion, str):
                target_emotion = image_or_emotion
                image = None
            else:
                image = image_or_emotion
        else:
            image = image_or_emotion if not isinstance(image_or_emotion, str) else None

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if img_embeds is None:
            if image is None:
                raise ValueError("Either image or img_embeds must be supplied to get_emotion_prompt_embeds.")
            img_embeds = self.extract_clip_image_embeds(image)

        # Extract CLIP text embeddings for target emotion
        text_data = self.clip_processor(text=target_emotion, return_tensors="pt", padding="max_length")
        with torch.no_grad():
            raw_text = self.clip_model.get_text_features(input_ids=text_data["input_ids"].to(device))
            text_embeds = raw_text.pooler_output if hasattr(raw_text, "pooler_output") else raw_text
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(1)

            # Fuse with 12-layer Q-Former (76 learned query tokens + 1 text embed -> 77 tokens)
            normal_output = self.qformer(
                text_embeds=text_embeds.to(device, dtype=torch.float16),
                img_embeds=img_embeds.to(device, dtype=torch.float16),
            )
            if normal_output.shape[1] == 76:
                pad = torch.zeros(
                    (normal_output.shape[0], 1, normal_output.shape[2]),
                    dtype=normal_output.dtype,
                    device=normal_output.device,
                )
                normal_output = torch.cat([normal_output, pad], dim=1)

        return normal_output.to(device, dtype=torch.float16)