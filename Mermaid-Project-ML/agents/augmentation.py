import os
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
        3. Uses InstructPix2Pix with LCM to quickly generate an image where the subject exhibits the emotion (using 4 steps).
        4. Saves the generated images to disk and returns their paths.
    """
    print("--- [Agent: Augmentation] Generating Emotion References ---")
    
    # 1. Ensure Diffusion Model is on GPU (This will offload the MLLM!)
    memory_manager.load_diffusion()
    pipeline = memory_manager.diffusion_pipeline
    
    # 2. Load the original image
    original_image = Image.open(state['image_path']).convert("RGB")
    
    generated_refs = {}
    
    # Ensure a directory exists for generated images
    os.makedirs("generated_refs", exist_ok=True)
    
    # 3. Generate images using LCM accelerated InstructPix2Pix
    for emotion in state['candidate_emotions']:
        # Format a prompt for InstructPix2Pix
        prompt = f"make them look {emotion}, express {emotion}"
        
        # LCM requires fewer steps and a low guidance scale
        with torch.no_grad():
            result_image = pipeline(
                prompt=prompt,
                image=original_image,
                num_inference_steps=4,
                guidance_scale=1.5,
                image_guidance_scale=1.5,
                output_type="pil"
            ).images[0]
            
        save_path = f"generated_refs/ref_{emotion.lower().replace(' ', '_')}.jpg"
        result_image.save(save_path)
        generated_refs[emotion] = save_path
        print(f"    -> Generated ref for '{emotion}' at {save_path}")
        
    return {"reference_images": generated_refs}
