import os
import gc
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator

def run_augmentation_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Generates reference images for ALL candidate emotions using a diffusion model.
    
    Input:
        - state: The current LangGraph state containing 'image_path' and 'candidate_emotions'.
        - memory_manager: Handles GPU/CPU model swapping and provides access to the diffusion pipeline.
        
    Output:
        - dict: A dictionary containing 'reference_images' mapping each emotion to its generated image path.
        
    What it does:
        1. Ensures the Diffusion Model is on GPU (offloads MLLM).
        2. Loops through each candidate emotion.
        3. Uses the fused Hyper-SD 4-step LoRA with Stable Diffusion 1.5 Img2Img to edit the image while preserving its structure.
        4. Saves the generated images to disk and returns their paths.
    """
    print("--- [Agent: Augmentation] Generating Emotion References ---")
    
    # 1. Ensure Diffusion Model is on GPU (This will offload the MLLM!)
    memory_manager.load_diffusion()
    pipeline = memory_manager.diffusion_pipeline
    
    generated_refs = {}
    
    # Ensure a directory exists for generated images
    os.makedirs("generated_refs", exist_ok=True)
    
    try:
        # 3. Generate structure-preserving emotion edits with Hyper-SD Img2Img.
        with Image.open(state['image_path']) as source_image:
            original_image = source_image.convert("RGB")

        for emotion in state['candidate_emotions']:
            caption = state.get('caption', '')
            prompt = f"{caption}, conveying a strong sense of {emotion.lower()}, highly detailed"

            with torch.no_grad():
                result_image = pipeline(
                    prompt=prompt,
                    image=original_image,
                    num_inference_steps=4,
                    guidance_scale=1.5,
                    strength=0.35,
                    output_type="pil",
                ).images[0].convert("RGB")

            save_path = f"generated_refs/ref_{emotion.lower().replace(' ', '_')}.jpg"
            result_image.save(save_path, quality=95)
            generated_refs[emotion] = save_path
            print(f"    -> Generated ref for '{emotion}' at {save_path}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    return {"reference_images": generated_refs}
