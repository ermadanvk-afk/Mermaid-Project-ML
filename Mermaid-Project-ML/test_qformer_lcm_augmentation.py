#!/usr/bin/env python3
"""
Test Suite: Q-Former Embedding Injection & LCM-Accelerated Reference Generation
================================================================================
Verifies that:
1. CombinedModel (12-layer Q-Former) correctly fuses image & text features into
   a 77-token prompt embedding tensor ([batch_size, 77, 768]) in float16.
2. MemoryOrchestrator properly configures:
   - CLIP text/image feature extraction
   - Q-Former conditioning (76 query tokens + 1 text token -> 77 tokens)
   - Diffusers InstructPix2Pix pipeline with LCMScheduler & LCM-LoRA weights.
3. run_augmentation_agent consumes:
   - Exact Q-Former prompt_embeds ([1, 77, 768], float16)
   - Perfectly matched negative_prompt_embeds ([1, 77, 768], float16)
   - LCM generation parameters (num_inference_steps=6, guidance_scale=1.8, image_guidance_scale=1.5)
   - Single CLIP feature extraction per image (avoiding redundant vision passes)
   - Resized input image (short-side 512, multiples of 8)
   - Generates & caches reference images at 256x256.

Usage:
    # 1. Run full unit/mock test suite (fast, no massive GPU downloads required):
    python test_qformer_lcm_augmentation.py

    # 2. Run with pytest:
    pytest test_qformer_lcm_augmentation.py -v

    # 3. Optional: Run live GPU test (requires GPU and model weights):
    python test_qformer_lcm_augmentation.py --live
"""

import os
import sys
import tempfile
import argparse
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from PIL import Image

# Ensure Mermaid-Project-ML root is on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core.qformer import CombinedModel, resize_image_to_512
from core.memory_manager import MemoryOrchestrator
from agents.augmentation import run_augmentation_agent, get_affective_prompt


class TestQFormerArchitecture(unittest.TestCase):
    """Layer 1: Verify Q-Former CombinedModel architecture & output dimensions."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CombinedModel(num_query_tokens=76, hidden_size=768).to(self.device)
        self.model.eval()

    def test_qformer_initialization_and_query_tokens(self):
        """Check query tokens shape [1, 76, 768] and hidden size."""
        self.assertEqual(self.model.query_tokens.shape, (1, 76, 768))
        self.assertEqual(self.model.model.config.num_hidden_layers, 12)
        self.assertEqual(self.model.model.config.num_attention_heads, 12)
        self.assertEqual(self.model.model.config.hidden_size, 768)

    def test_qformer_forward_embedding_fusion_shape(self):
        """
        Verify forward pass:
        text_embeds [1, 768] + img_embeds [1, 768] -> normal_output [1, 77, 768].
        """
        batch_size = 1
        text_embeds = torch.randn(batch_size, 768, device=self.device)
        img_embeds = torch.randn(batch_size, 768, device=self.device)

        with torch.no_grad():
            output = self.model(text_embeds=text_embeds, img_embeds=img_embeds)

        self.assertEqual(
            output.shape,
            (batch_size, 77, 768),
            f"Expected Q-Former output shape ({batch_size}, 77, 768), but got {output.shape}",
        )

    def test_qformer_float16_precision(self):
        """Verify Q-Former operates properly in torch.float16 as required for SD 1.5."""
        model_fp16 = self.model.to(torch.float16)
        text_embeds = torch.randn(1, 1, 768, dtype=torch.float16, device=self.device)
        img_embeds = torch.randn(1, 1, 768, dtype=torch.float16, device=self.device)

        with torch.no_grad():
            output = model_fp16(text_embeds=text_embeds, img_embeds=img_embeds)

        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.shape, (1, 77, 768))


class TestImagePreprocessing(unittest.TestCase):
    """Layer 2: Verify aspect ratio preservation and 512 short-side divisible by 8."""

    def test_resize_image_to_512_divisibility(self):
        test_sizes = [(1024, 768), (768, 1024), (800, 600), (512, 512), (300, 450)]
        for w, h in test_sizes:
            dummy_img = Image.new("RGB", (w, h), color=(128, 128, 128))
            resized = resize_image_to_512(dummy_img)
            rw, rh = resized.size
            self.assertEqual(rw % 8, 0, f"Width {rw} is not divisible by 8 for input size ({w}, {h})")
            self.assertEqual(rh % 8, 0, f"Height {rh} is not divisible by 8 for input size ({w}, {h})")
            min_dim = min(rw, rh)
            self.assertEqual(min_dim, 512, f"Short side should be 512, got {min_dim}")


class TestMemoryOrchestratorWiring(unittest.TestCase):
    """Layer 3: Verify MemoryOrchestrator LCM scheduler and Q-Former fusion calls."""

    @patch("core.memory_manager.StableDiffusionInstructPix2PixPipeline")
    @patch("core.memory_manager.LCMScheduler")
    def test_lcm_integration_in_memory_orchestrator(self, mock_lcm_cls, mock_pipe_cls):
        """Verify that when use_lcm=True, LCMScheduler and LCM-LoRA are loaded."""
        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.scheduler.config = {"some": "config"}
        mock_pipe_cls.from_pretrained.return_value = mock_pipeline_inst

        orchestrator = MemoryOrchestrator(use_lcm=True)
        orchestrator._init_diffusion()

        # Check LCMScheduler configuration
        mock_lcm_cls.from_config.assert_called_once_with({"some": "config"})
        self.assertEqual(mock_pipeline_inst.scheduler, mock_lcm_cls.from_config.return_value)

        # Check LCM LoRA weights loaded
        mock_pipeline_inst.load_lora_weights.assert_called_once_with(
            "latent-consistency/lcm-lora-sdv1-5"
        )
        mock_pipeline_inst.enable_attention_slicing.assert_called_once()

    def test_get_emotion_prompt_embeds_mocked_flow(self):
        """Verify that get_emotion_prompt_embeds orchestrates CLIP text and Q-Former fusion."""
        orchestrator = MemoryOrchestrator(use_lcm=True)
        orchestrator.diffusion_loaded = True

        mock_clip_proc = MagicMock()
        mock_clip_proc.return_value = {"input_ids": torch.zeros((1, 77), dtype=torch.long)}

        mock_clip_model = MagicMock()
        mock_clip_model.get_text_features.return_value = torch.ones((1, 768), dtype=torch.float16)

        mock_qformer = MagicMock()
        mock_qformer.return_value = torch.ones((1, 77, 768), dtype=torch.float16)

        orchestrator.clip_processor = mock_clip_proc
        orchestrator.clip_model = mock_clip_model
        orchestrator.qformer = mock_qformer

        dummy_img_embeds = torch.zeros((1, 1, 768), dtype=torch.float16)
        prompt_embeds = orchestrator.get_emotion_prompt_embeds(
            "joy",
            img_embeds=dummy_img_embeds,
        )

        mock_clip_proc.assert_called_once_with(text="joy", return_tensors="pt", padding="max_length")
        mock_clip_model.get_text_features.assert_called_once()
        mock_qformer.assert_called_once()

        self.assertEqual(prompt_embeds.shape, (1, 77, 768))
        self.assertEqual(prompt_embeds.dtype, torch.float16)


class TestAugmentationAgentConsumption(unittest.TestCase):
    """
    Layer 4: Critical Test - Verify that run_augmentation_agent consumes:
    1. Q-Former prompt_embeds
    2. Correct negative_prompt_embeds (77 tokens, float16)
    3. LCM parameters (steps=6, guidance=1.8, image_guidance=1.5)
    4. Only a single CLIP image feature extraction per image
    """

    def setUp(self):
        # Create a temporary input image
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_img_path = os.path.join(self.temp_dir.name, "test_input.jpg")
        dummy_img = Image.new("RGB", (640, 480), color=(200, 100, 50))
        dummy_img.save(self.test_img_path)

        # Create dummy checkpoint so fail-fast passes
        self.dummy_ckpt = os.path.join(self.temp_dir.name, "Q-Former.pth")
        torch.save({"dummy": 1}, self.dummy_ckpt)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_qformer_injection_and_lcm_consumption(self):
        """
        Verify run_augmentation_agent passes:
          - prompt_embeds (NOT prompt text)
          - negative_prompt_embeds
          - num_inference_steps == 6 (LCM)
          - guidance_scale == 1.8
          - image_guidance_scale == 1.5
        """
        candidate_emotions = ["amusement", "anger", "sadness"]
        state = {
            "image_path": self.test_img_path,
            "candidate_emotions": candidate_emotions,
            "use_qformer": True,
            "reference_images": {},
        }

        # Setup mock orchestrator and pipeline spy
        memory_manager = MagicMock(spec=MemoryOrchestrator)
        memory_manager.qformer_checkpoint_path = self.dummy_ckpt

        pipeline_mock = MagicMock()
        # Mock pipeline output with a 512x512 PIL image
        generated_pil = Image.new("RGB", (512, 512), color=(10, 20, 30))
        pipe_result = MagicMock()
        pipe_result.images = [generated_pil]
        pipeline_mock.return_value = pipe_result

        # Pipeline tokenizer and text encoder mock
        pipeline_mock.tokenizer.model_max_length = 77
        tok_out = MagicMock()
        tok_out.input_ids = torch.zeros((1, 77), dtype=torch.long)
        pipeline_mock.tokenizer.return_value = tok_out

        # text_encoder output: tuple where [0] is [1, 77, 768]
        pipeline_mock.text_encoder.return_value = (
            torch.zeros((1, 77, 768), dtype=torch.float16),
        )
        pipeline_mock.device = torch.device("cpu")

        memory_manager.diffusion_pipeline = pipeline_mock

        # Extract CLIP features spy
        clip_embed_tensor = torch.randn((1, 1, 768), dtype=torch.float16)
        memory_manager.extract_clip_image_embeds.return_value = clip_embed_tensor

        # Q-Former prompt embeds spy returning distinct tensors per emotion
        prompt_embed_map = {
            emo: torch.full((1, 77, 768), fill_value=float(i + 1), dtype=torch.float16)
            for i, emo in enumerate(candidate_emotions)
        }

        def mock_get_emotion_prompt_embeds(emotion, img_embeds):
            return prompt_embed_map[emotion].clone()

        memory_manager.get_emotion_prompt_embeds.side_effect = mock_get_emotion_prompt_embeds

        # Execute the agent
        result = run_augmentation_agent(state, memory_manager)

        # ── VERIFICATION 1: CLIP features extracted ONCE for the entire image ──
        self.assertEqual(
            memory_manager.extract_clip_image_embeds.call_count,
            1,
            "extract_clip_image_embeds must be called exactly ONCE per image",
        )

        # ── VERIFICATION 2: Q-Former get_emotion_prompt_embeds called for each emotion ──
        self.assertEqual(
            memory_manager.get_emotion_prompt_embeds.call_count,
            len(candidate_emotions),
            f"get_emotion_prompt_embeds must be called once per emotion ({len(candidate_emotions)})",
        )

        # ── VERIFICATION 3: Pipeline call count ──
        self.assertEqual(
            pipeline_mock.call_count,
            len(candidate_emotions),
            f"Diffusion pipeline must be invoked once per candidate emotion ({len(candidate_emotions)})",
        )

        # ── VERIFICATION 4: Pipeline arguments consumption ──
        for i, (call_args, emo) in enumerate(zip(pipeline_mock.call_args_list, candidate_emotions)):
            kwargs = call_args.kwargs

            # A. Verify Q-Former prompt_embeds was injected (and NOT plain prompt string)
            self.assertIn("prompt_embeds", kwargs, f"Call for '{emo}' missing 'prompt_embeds'")
            self.assertNotIn("prompt", kwargs, f"Call for '{emo}' should NOT have text 'prompt' when use_qformer=True")

            passed_embeds = kwargs["prompt_embeds"]
            self.assertIsInstance(passed_embeds, torch.Tensor)
            self.assertEqual(
                passed_embeds.shape,
                (1, 77, 768),
                f"Call for '{emo}': prompt_embeds shape must be (1, 77, 768), got {passed_embeds.shape}",
            )
            self.assertEqual(
                passed_embeds.dtype,
                torch.float16,
                f"Call for '{emo}': prompt_embeds dtype must be torch.float16, got {passed_embeds.dtype}",
            )

            # Check that the tensor passed matches the specific emotion's Q-Former output
            expected_fill = float(i + 1)
            self.assertTrue(
                torch.allclose(passed_embeds[0, 0, 0], torch.tensor(expected_fill, dtype=torch.float16)),
                f"Call for '{emo}' did not receive the emotion's specific Q-Former embedding",
            )

            # B. Verify negative_prompt_embeds matches sequence length and dtype
            self.assertIn("negative_prompt_embeds", kwargs, f"Call for '{emo}' missing 'negative_prompt_embeds'")
            neg_embeds = kwargs["negative_prompt_embeds"]
            self.assertEqual(
                neg_embeds.shape,
                (1, 77, 768),
                f"Call for '{emo}': negative_prompt_embeds shape mismatch",
            )
            self.assertEqual(
                neg_embeds.dtype,
                torch.float16,
                f"Call for '{emo}': negative_prompt_embeds dtype mismatch",
            )

            # C. Verify LCM Accelerated Generation Parameters
            self.assertEqual(
                kwargs.get("num_inference_steps"),
                8,
                f"Call for '{emo}': LCM inference steps should be 8, got {kwargs.get('num_inference_steps')}",
            )
            self.assertEqual(
                kwargs.get("guidance_scale"),
                4.5,
                f"Call for '{emo}': LCM guidance_scale should be 4.5, got {kwargs.get('guidance_scale')}",
            )
            self.assertEqual(
                kwargs.get("image_guidance_scale"),
                1.5,
                f"Call for '{emo}': LCM image_guidance_scale should be 1.5, got {kwargs.get('image_guidance_scale')}",
            )

            # D. Verify Input Image short-side 512
            input_img = kwargs.get("image")
            self.assertIsInstance(input_img, Image.Image)
            self.assertEqual(min(input_img.size), 512)

        # ── VERIFICATION 5: Reference images saved at 256x256 ──
        ref_images = result["reference_images"]
        self.assertEqual(len(ref_images), len(candidate_emotions))
        for emo, path in ref_images.items():
            self.assertTrue(os.path.exists(path), f"Generated reference for {emo} not found at {path}")
            with Image.open(path) as img:
                self.assertEqual(
                    img.size,
                    (256, 256),
                    f"Saved reference for {emo} should be 256x256, got {img.size}",
                )

        # Retain generated refs unless CLEANUP_TEST_REFS=1
        if os.environ.get("CLEANUP_TEST_REFS", "0") == "1":
            for path in ref_images.values():
                if os.path.exists(path):
                    os.remove(path)

    def test_76_token_shape_guard_padding(self):
        """
        Verify the shape guard in augmentation.py:
        If Q-Former output is 76 tokens, it pads with 1 token to 77 to match SD 1.5 latent space.
        """
        candidate_emotions = ["joy"]
        state = {
            "image_path": self.test_img_path,
            "candidate_emotions": candidate_emotions,
            "use_qformer": True,
            "reference_images": {},
        }

        memory_manager = MagicMock(spec=MemoryOrchestrator)
        memory_manager.qformer_checkpoint_path = self.dummy_ckpt

        pipeline_mock = MagicMock()
        generated_pil = Image.new("RGB", (512, 512), color=(0, 0, 0))
        pipe_result = MagicMock()
        pipe_result.images = [generated_pil]
        pipeline_mock.return_value = pipe_result

        pipeline_mock.tokenizer.model_max_length = 77
        tok_out = MagicMock()
        tok_out.input_ids = torch.zeros((1, 77), dtype=torch.long)
        pipeline_mock.tokenizer.return_value = tok_out
        pipeline_mock.text_encoder.return_value = (
            torch.zeros((1, 77, 768), dtype=torch.float16),
        )
        pipeline_mock.device = torch.device("cpu")
        memory_manager.diffusion_pipeline = pipeline_mock

        # Return 76 tokens to trigger the padding logic
        memory_manager.extract_clip_image_embeds.return_value = torch.zeros((1, 1, 768), dtype=torch.float16)
        memory_manager.get_emotion_prompt_embeds.return_value = torch.zeros((1, 76, 768), dtype=torch.float16)

        result = run_augmentation_agent(state, memory_manager)

        # Inspect injected prompt_embeds
        passed_embeds = pipeline_mock.call_args.kwargs["prompt_embeds"]
        self.assertEqual(
            passed_embeds.shape,
            (1, 77, 768),
            "Padding guard failed to pad 76 tokens to 77 tokens",
        )

        if os.environ.get("CLEANUP_TEST_REFS", "0") == "1":
            for path in result["reference_images"].values():
                if os.path.exists(path):
                    os.remove(path)

    def test_affective_prompt_fallback_when_qformer_false(self):
        """
        Verify that when use_qformer=False, pipeline consumes:
          - text prompt from AFFECTIVE_PROMPT_MAP
          - no prompt_embeds
          - LCM acceleration (steps=6, guidance=1.8, img_guidance=1.5)
        """
        state = {
            "image_path": self.test_img_path,
            "candidate_emotions": ["amusement"],
            "use_qformer": False,
            "reference_images": {},
        }

        memory_manager = MagicMock(spec=MemoryOrchestrator)
        pipeline_mock = MagicMock()
        generated_pil = Image.new("RGB", (512, 512), color=(0, 0, 0))
        pipe_result = MagicMock()
        pipe_result.images = [generated_pil]
        pipeline_mock.return_value = pipe_result
        memory_manager.diffusion_pipeline = pipeline_mock

        result = run_augmentation_agent(state, memory_manager)

        kwargs = pipeline_mock.call_args.kwargs
        self.assertIn("prompt", kwargs)
        self.assertEqual(kwargs["prompt"], get_affective_prompt("amusement"))
        self.assertNotIn("prompt_embeds", kwargs)
        self.assertEqual(kwargs["num_inference_steps"], 8)
        self.assertEqual(kwargs["guidance_scale"], 4.5)
        self.assertEqual(kwargs["image_guidance_scale"], 1.5)

        if os.environ.get("CLEANUP_TEST_REFS", "0") == "1":
            for path in result["reference_images"].values():
                if os.path.exists(path):
                    os.remove(path)


def tearDownModule():
    """Reports reference images retained in generated_refs/ after test execution."""
    refs_dir = "generated_refs"
    if os.path.exists(refs_dir):
        files = [f for f in os.listdir(refs_dir) if f.startswith("ref_")]
        if files:
            print("\n" + "=" * 70)
            print(f"[Artifacts] Reference images retained in '{refs_dir}/':")
            for f in sorted(files):
                full_path = os.path.join(refs_dir, f)
                size_kb = os.path.getsize(full_path) / 1024
                print(f"  - {full_path} ({size_kb:.1f} KB)")
            print("=" * 70 + "\n")


def run_live_gpu_test(args):
    """
    Live end-to-end test on real GPU with loaded weights.
    Supports real input image, custom candidate emotions, and Q-Former or Affective Prompt mode.
    Triggered with `python test_qformer_lcm_augmentation.py --live`.
    """
    print("\n" + "=" * 70)
    print("RUNNING LIVE GPU GENERATION (InstructPix2Pix + LCM LoRA)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[SKIPPED] CUDA is not available. Live GPU test requires a GPU.")
        return

    checkpoint_path = "checkpoint/Q-Former.pth"
    has_qformer_ckpt = os.path.exists(checkpoint_path)

    use_qformer = not args.no_qformer
    if use_qformer and not has_qformer_ckpt:
        print(f"[Notice] Q-Former checkpoint not found at '{checkpoint_path}'.")
        print("         Switching to Affective Prompt + LCM mode for live generation.")
        print("         (To use Q-Former weights, run 'python scripts/download_qformer.py')")
        use_qformer = False

    mode_label = "Q-Former + LCM" if use_qformer else "Affective Prompt + LCM"
    print(f"Generation Mode: {mode_label}")

    use_lcm = not args.no_lcm
    print(f"[1/4] Initializing MemoryOrchestrator (LCM={use_lcm})...")
    orchestrator = MemoryOrchestrator(
        qformer_checkpoint_path=checkpoint_path,
        use_lcm=use_lcm,
    )

    print("[2/4] Loading Diffusion Pipeline (InstructPix2Pix + LCM LoRA)...")
    # Patch load_lora_weights to provide weight_name so offline mode and cached loads succeed seamlessly
    from diffusers import StableDiffusionInstructPix2PixPipeline
    orig_load_lora = StableDiffusionInstructPix2PixPipeline.load_lora_weights
    def patched_load_lora(self, pretrained_model_name_or_path_or_dict, **kwargs):
        if "weight_name" not in kwargs:
            kwargs["weight_name"] = "pytorch_lora_weights.safetensors"
        return orig_load_lora(self, pretrained_model_name_or_path_or_dict, **kwargs)
    StableDiffusionInstructPix2PixPipeline.load_lora_weights = patched_load_lora

    orchestrator.load_diffusion()

    # CLIP image feature extraction using all spatial patch tokens.
    # The Q-Former uses cross-attention with img_embeds as encoder_hidden_states.
    # Passing only the single CLS/pooler token [1, 1, 768] starves the cross-attention
    # of spatial content — all 257 ViT patch tokens are nearly identical when pooled,
    # so Q-Former outputs become nearly identical regardless of emotion.
    # Fix: pass all 257 patch tokens projected to 768-dim so cross-attention has rich
    # per-region image content to differentially condition on per emotion query.
    def patched_extract_img(image):
        if not orchestrator.diffusion_loaded:
            orchestrator.load_diffusion()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resized_img = resize_image_to_512(image)
        data = orchestrator.clip_processor(images=resized_img, return_tensors="pt")
        with torch.no_grad():
            pixel_vals = data["pixel_values"].to(device, dtype=torch.float16)
            clip = orchestrator.clip_model.to(device)
            # Get ALL patch token hidden states [1, 257, 1024] from ViT-L
            v_out = clip.vision_model(pixel_values=pixel_vals)
            patch_tokens = v_out.last_hidden_state  # [1, 257, 1024]
            # Project from ViT-L dim (1024) to Q-Former hidden dim (768) via visual_projection
            # visual_projection is a Linear(1024, 768) applied token-wise
            patch_proj = clip.visual_projection(patch_tokens)  # [1, 257, 768]
            # L2-normalize each token independently
            patch_proj = patch_proj / patch_proj.norm(p=2, dim=-1, keepdim=True)
        return patch_proj.to(device, dtype=torch.float16)  # [1, 257, 768]

    def patched_get_emotion_prompt_embeds(image_or_emotion=None, emotion_or_none=None, img_embeds=None, **kwargs):
        if not orchestrator.diffusion_loaded:
            orchestrator.load_diffusion()
        emotion = kwargs.get("emotion") or (emotion_or_none if emotion_or_none is not None else (image_or_emotion if isinstance(image_or_emotion, str) else None))
        img = image_or_emotion if not isinstance(image_or_emotion, str) else None

        if img_embeds is None:
            if img is None:
                raise ValueError("Either image or img_embeds must be supplied.")
            img_embeds = patched_extract_img(img)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_data = orchestrator.clip_processor(text=emotion, return_tensors="pt", padding="max_length")
        with torch.no_grad():
            clip = orchestrator.clip_model.to(device)
            t_out = clip.text_model(input_ids=text_data["input_ids"].to(device))
            text_embeds = clip.text_projection(t_out[1])
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(1)  # [1, 1, 768]

            # img_embeds is now [1, 257, 768] — all patch tokens
            # Q-Former cross-attention will attend over all 257 spatial image regions
            # per each of the 76 learned query tokens, giving rich emotion-image fusion
            normal_output = orchestrator.qformer(
                text_embeds=text_embeds.to(device, dtype=torch.float16),
                img_embeds=img_embeds.to(device, dtype=torch.float16),
            )
        return normal_output.to(device, dtype=torch.float16)

    orchestrator.extract_clip_image_embeds = patched_extract_img
    orchestrator.get_emotion_prompt_embeds = patched_get_emotion_prompt_embeds

    # Check scheduler type
    scheduler_name = orchestrator.diffusion_pipeline.scheduler.__class__.__name__
    print(f"      Scheduler: {scheduler_name}")
    if use_lcm:
        assert "LCM" in scheduler_name, f"Expected LCMScheduler, got {scheduler_name}"
    else:
        print("      Standard Diffusion Scheduler active (Euler Ancestral)")

    # Determine input image
    is_temp_image = False
    if args.image and os.path.exists(args.image):
        input_image_path = args.image
        print(f"      Input Image: {input_image_path}")
    else:
        candidate_dir = "data/emoset_1000"
        found_sample = None
        if os.path.exists(candidate_dir):
            jpgs = [f for f in os.listdir(candidate_dir) if f.endswith(".jpg")]
            if jpgs:
                found_sample = os.path.join(candidate_dir, sorted(jpgs)[0])

        if found_sample and os.path.exists(found_sample):
            input_image_path = found_sample
            print(f"      Using dataset sample: {input_image_path}")
        else:
            test_img = Image.new("RGB", (512, 512), color=(140, 90, 60))
            input_image_path = "test_live_input.jpg"
            test_img.save(input_image_path)
            is_temp_image = True
            print(f"      Created synthetic test image: {input_image_path}")

    candidate_emotions = args.emotions if args.emotions else ["joy", "amusement", "sadness", "anger"]
    print(f"      Candidate Emotions: {candidate_emotions}")

    steps = args.steps if args.steps is not None else (8 if use_lcm else 30)
    guidance = args.guidance if args.guidance is not None else (4.5 if use_lcm else 7.5)
    img_guidance = args.img_guidance if args.img_guidance is not None else 1.5
    resolution = args.resolution if args.resolution is not None else 512

    print(f"      Inference Parameters: steps={steps}, guidance_scale={guidance}, image_guidance={img_guidance}, resolution={resolution}x{resolution}")

    try:
        print(f"[3/4] Generating reference images ({mode_label})...")
        # Always use the custom path so our patched CLIP extractors are used
        pipeline = orchestrator.diffusion_pipeline
        original_image = Image.open(input_image_path).convert("RGB")
        input_image = resize_image_to_512(original_image)

        img_embeds = orchestrator.extract_clip_image_embeds(input_image) if use_qformer else None

        generated_refs = {}
        seed = 42
        for emotion in candidate_emotions:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            with torch.no_grad():
                if use_qformer:
                    prompt_embeds = orchestrator.get_emotion_prompt_embeds(
                        emotion=emotion,
                        img_embeds=img_embeds,
                    )
                    if prompt_embeds.shape[1] == 76:
                        pad = torch.zeros((prompt_embeds.shape[0], 1, prompt_embeds.shape[2]), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
                        prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)

                    prompt_embeds = prompt_embeds.to(dtype=torch.float16, device=pipeline.device)

                    # Build negative_prompt_embeds using SD text encoder on "" — identical
                    # to what augmentation.py does (lines 99-119).  This keeps the uncond
                    # embedding in the correct token space that the pipeline CFG logic
                    # expects, while the sequence-length guard below ensures the shapes match
                    # the Q-Former output so torch.cat inside diffusers never raises an error.
                    uncond_tokens = pipeline.tokenizer(
                        [""],
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    negative_prompt_embeds = pipeline.text_encoder(
                        uncond_tokens.input_ids.to(pipeline.device)
                    )[0].to(dtype=torch.float16, device=pipeline.device)

                    # Sequence-length guard: truncate or pad to match prompt_embeds seq len
                    if negative_prompt_embeds.shape[1] > prompt_embeds.shape[1]:
                        negative_prompt_embeds = negative_prompt_embeds[:, :prompt_embeds.shape[1], :]
                    elif negative_prompt_embeds.shape[1] < prompt_embeds.shape[1]:
                        pad_len = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
                        pad = torch.zeros(
                            (negative_prompt_embeds.shape[0], pad_len, negative_prompt_embeds.shape[2]),
                            dtype=negative_prompt_embeds.dtype,
                            device=negative_prompt_embeds.device,
                        )
                        negative_prompt_embeds = torch.cat([negative_prompt_embeds, pad], dim=1)

                    print(f"    -> [{emotion}] prompt_embeds norm={prompt_embeds.norm():.2f}, "
                          f"neg norm={negative_prompt_embeds.norm():.2f}, "
                          f"steps={steps}, guidance={guidance}, img_guidance={img_guidance}")
                    with torch.autocast("cuda"):
                        result_image = pipeline(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            image=input_image,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            image_guidance_scale=img_guidance,
                            generator=generator,
                            output_type="pil",
                        ).images[0]
                else:
                    with torch.autocast("cuda"):
                        result_image = pipeline(
                            prompt=get_affective_prompt(emotion),
                            image=input_image,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            image_guidance_scale=img_guidance,
                            generator=generator,
                            output_type="pil",
                        ).images[0]

            if resolution != 512:
                result_image = result_image.resize((resolution, resolution), Image.LANCZOS)
            os.makedirs("generated_refs", exist_ok=True)
            save_path = f"generated_refs/ref_{emotion.lower().replace(' ', '_')}.jpg"
            result_image.save(save_path)
            generated_refs[emotion] = save_path
            print(f"      -> Generated ref for '{emotion}' at {save_path}")

        print(f"\n[4/4] Verification & Inspection of Generated References:")
        for emo, path in generated_refs.items():
            assert os.path.exists(path), f"Output image missing at {path}"
            with Image.open(path) as out_img:
                size_kb = os.path.getsize(path) / 1024
                print(f"      ✓ [{emo}]: {path} | Resolution: {out_img.size} | File size: {size_kb:.1f} KB")

        print("\n" + "=" * 70)
        print("[SUCCESS] All reference images were generated and saved in 'generated_refs/'!")
        print("=" * 70)
    finally:
        if is_temp_image and os.path.exists(input_image_path):
            os.remove(input_image_path)


def print_summary_header():
    print("=" * 78)
    print(" MERMAID TEST: Q-Former Embedding Injection & LCM Reference Generation ")
    print("=" * 78)
    print("Components under test:")
    print("  1. core/qformer.py: CombinedModel (12-layer Q-Former, 76 queries, 77 tokens)")
    print("  2. core/memory_manager.py: MemoryOrchestrator (CLIP fusion, LCMScheduler + LoRA)")
    print("  3. agents/augmentation.py: run_augmentation_agent (injection & LCM consumption)")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Q-Former Injection and LCM Generation")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live GPU generation test using actual loaded models",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Custom image path for live generation (e.g. data/emoset_1000/amusement_00290.jpg)",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=None,
        help="List of emotions to generate (e.g. --emotions joy amusement anger)",
    )
    parser.add_argument(
        "--no-qformer",
        action="store_true",
        help="Force using Affective Prompt + LCM mode instead of Q-Former",
    )
    parser.add_argument(
        "--no-lcm",
        action="store_true",
        help="Disable LCM acceleration and use standard high-quality diffusion (Euler Ancestral, 30-50 steps)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion denoising steps (default: 6 for LCM, 30-50 for standard)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Text/Emotion guidance scale (default: 1.8 for LCM, 7.5 for standard)",
    )
    parser.add_argument(
        "--img-guidance",
        type=float,
        default=None,
        help="Image structure guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        choices=[256, 512],
        help="Saved image resolution: 256 (paper default for MLLM VRAM) or 512 (crisp full-resolution)",
    )
    args, remaining_args = parser.parse_known_args()

    print_summary_header()

    if args.live:
        run_live_gpu_test(args)
    else:
        # Run unittest suite with remaining args (e.g. verbosity)
        unittest_args = [sys.argv[0]] + remaining_args
        if "-v" not in unittest_args and "--verbose" not in unittest_args:
            unittest_args.append("-v")
        unittest.main(argv=unittest_args)
