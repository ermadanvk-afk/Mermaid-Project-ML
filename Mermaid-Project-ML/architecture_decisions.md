# MERMAID Implementation: Architectural Decisions & Technical Adaptations

## 1. Hardware Constraints & Memory Orchestration (12GB VRAM Limit)
**The Problem:** The MERMAID paper assumes access to massive compute (e.g., NVIDIA H100s). We were constrained to a single 12GB VRAM GPU. Running a 7B parameter Multimodal LLM (Qwen2.5-VL) and a Stable Diffusion model concurrently on 12GB natively results in instant Out-Of-Memory (OOM) failures.
**The Solution:** We engineered a custom `MemoryOrchestrator` (`core/memory_manager.py`). 
*   **4-bit Quantization:** The MLLM is loaded in 4-bit (NF4) using `bitsandbytes`, shrinking its footprint from ~14GB to ~5GB.
*   **Dynamic VRAM Clearing:** The orchestrator aggressively flushes the KV cache (`torch.cuda.empty_cache()`) and garbage-collects tensors when the LangGraph shifts from the MLLM nodes to the Diffusion node. 
*   **Resolution Capping:** Qwen2.5-VL uses dynamic resolution. High-res images from the Emotion6 dataset generated tens of thousands of tokens, crashing the GPU. We aligned with the paper's preprocessing by enforcing `max_pixels=65536` (256x256), capping sequence length and preserving VRAM.

## 2. Accelerating the Augmentation Loop (LCM Integration)
**The Problem:** The paper's authors used InstructPix2Pix with 100 denoising steps per emotion. In an iterative LangGraph loop testing 6 to 8 emotions, this equates to 600-800 denoising steps per image, rendering the evaluation process impractically slow on consumer hardware.
**The Solution:** We integrated a **Latent Consistency Model (LCM) LoRA** (`latent-consistency/lcm-lora-sdv1-5`) into the `InstructPix2Pix` pipeline (`memory_manager.py`). By replacing the standard scheduler with `LCMScheduler`, we achieved equivalent visual edits in just **4 denoising steps** instead of 100, accelerating the augmentation agent by over 20x while maintaining structural integrity.

## 3. Simulating Proprietary Emotion Embeddings (The Q-Former Workaround)
**The Problem:** The paper explicitly states they used a custom, pretrained "Q-Former" (the Emotion Adapter from the *EmoEdit* paper) to inject emotion embeddings directly into the latent prompt space of InstructPix2Pix. Because these weights are not natively supported in standard Hugging Face `diffusers` pipelines, we could not replicate this step exactly.
**The Solution:** We simulated the Q-Former's intended behavior using natural language text prompts. We passed explicit English instructions (e.g., `prompt = f"alter the image to express a strong feeling of {emotion}"`) to the standard InstructPix2Pix text encoder. This allowed us to successfully recreate the emotion-conditioned references using purely open-source models.

## 4. Multi-Image Interleaved Prompting vs. Text Captions
**The Problem:** Early in the project, we used LLaVA-NeXT, which suffered from a strict architectural limitation: it could only process one image per prompt. To provide the Visual Reflection agent with the 6-8 augmented reference images, we had to run a loop generating text captions for every single image and feed the captions to the LLM.
**The Solution:** We migrated the core MLLM to **Qwen2.5-VL-7B-Instruct**. Qwen natively supports interleaved multi-image inputs. We refactored `visual_reflection.py` to completely eliminate the captioning hack. The agent now receives the Original Query Image and ALL Augmented Reference Images directly in its vision encoder simultaneously, drastically improving both reasoning accuracy (it can actually *see* the references) and evaluation speed.

## 5. Overcoming LLM "Laziness" (Unanimous Voting)
**The Problem:** During testing, we noticed the system was skipping the visual reflection loop almost entirely. The MLLM, acting as the Text Reflection agent, was "lazy" and overly agreeable, routinely voting `good_enough` across all 3 perspectives based solely on the text caption.
**The Solution:** We tightened the `_aggregate_votes` logic in `agents/text_reflection.py` to require **Unanimous Voting**. If even a *single* perspective (out of 3) spots a discrepancy and votes `needs_revision`, the graph forces the pipeline into the Augmentation and Visual Reflection loop. This successfully mirrors the paper's philosophy of "third-party conflict resolution," forcing the system to rely on visual evidence when text is ambiguous.
