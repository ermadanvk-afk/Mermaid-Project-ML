# MERMAID: Multi-Perspective Self-Reflection and Emotion-Guided Augmentation

Welcome to our implementation of the **MERMAID** framework! This codebase translates state-of-the-art affective image recognition research into a compute-efficient, open-source architecture capable of running on consumer hardware.

## 🎯 The Core Concept
Emotion recognition in images is inherently subjective and complex. Standard Vision-Language Models (VLMs) struggle because they only look at the image once, often misinterpreting subtle cues or hallucinating context. 

MERMAID solves this using a **LangGraph-powered Multi-Agent System** that mimics human critical thinking:
1. **Initial Assessment:** The system generates a caption and makes an initial guess.
2. **Text Reflection (The Skeptics):** Three distinct "perspectives" (e.g., Focus on Scene, Focus on Color, Default) critique the initial guess.
3. **Visual Augmentation (The Imagination):** If there is *any* disagreement among the perspectives, the system asks: *"What would this image look like if it were actually Sad? Or Angry?"* It generates those alternate realities using Diffusion models.
4. **Visual Reflection (The Comparison):** The MLLM looks at the original image alongside the generated "alternate realities" to confidently determine which emotion fits best.

## 🛠️ Our Implementation & Innovations
While the original paper relied on massive compute clusters (NVIDIA H100s) and proprietary models, our primary achievement is engineering this architecture to run efficiently on a **single 12GB VRAM GPU**.

### Key System Features
* **Qwen2.5-VL Integration:** Replaced the paper's baseline models with Qwen2.5-VL. We utilized Qwen's advanced interleaved multi-image capabilities to allow the Visual Reflection agent to process the original image and up to 8 augmented images *simultaneously* in a single pass.
* **LCM Accelerated Diffusion:** Generating 8 augmented reference images using standard Stable Diffusion (100 steps) takes minutes per image. By integrating a Latent Consistency Model (LCM) LoRA, we achieved high-quality emotion augmentation in just **4 denoising steps**, speeding up the visual loop by over 20x.
* **Intelligent Memory Orchestration:** We built a custom `MemoryOrchestrator` that hot-swaps the 4-bit quantized MLLM and the FP16 Diffusion pipeline within the strict 12GB VRAM constraint, dynamically clearing KV caches and tensors to prevent memory fragmentation during massive dataset evaluations.
* **Dataset Agnostic:** Successfully evaluated across standard benchmarks including **EmoSet**, **ArtPhoto**, and the highly challenging, high-resolution **Emotion6** dataset.

## 🚀 How to Run the Code
The entire multi-agent loop is orchestrated via LangGraph. 
To start a full evaluation on the Emotion6 dataset:
```bash
python main.py
```
* **Inputs:** Raw internet images (e.g., Flickr scrapes from Emotion6).
* **Outputs:** A structured CSV log (`results/emotion6_evaluation_qwen2.5.csv`) tracking ground truth vs. predicted outcomes, along with detailed terminal traces of the agent debates.

## 📊 Conclusion
Our implementation successfully proves that complex, multi-agent self-reflection workflows for affective computing do not require enterprise-grade hardware. Through strategic model quantization, diffusion acceleration, and tight memory management, we have replicated the core philosophical successes of the MERMAID framework for the open-source community.
