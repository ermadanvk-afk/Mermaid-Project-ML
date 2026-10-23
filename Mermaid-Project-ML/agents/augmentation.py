import os
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator

def run_augmentation_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Generates reference images for candidate emotions using InstructPix2Pix with LCM.
    Caches reference images per input image so repeated reflection loops do not re-run diffusion.
    """
    img_stem = os.path.splitext(os.path.basename(state['image_path']))[0]
    cache_dir = os.path.join("generated_refs", img_stem)
    os.makedirs(cache_dir, exist_ok=True)
    
    existing_refs = state.get("reference_images", {})
    emotions_needed = [
        e for e in state["candidate_emotions"]
        if e not in existing_refs or not os.path.exists(existing_refs[e])
    ]
    
    if not emotions_needed:
        print(f"--- [Agent: Augmentation] Reusing all cached references for {img_stem} ---")
        return {"reference_images": existing_refs}
        
    print(f"--- [Agent: Augmentation] Generating References ({len(emotions_needed)} needed) ---")
    memory_manager.load_diffusion()
    pipeline = memory_manager.diffusion_pipeline
    original_image = Image.open(state['image_path']).convert("RGB").resize((512, 512))
    
    generated_refs = dict(existing_refs)
    
    for emotion in emotions_needed:
        save_path = os.path.join(cache_dir, f"ref_{emotion.lower().replace(' ', '_')}.jpg")
        if os.path.exists(save_path):
            generated_refs[emotion] = save_path
            print(f"    - Reusing cached ref for '{emotion}' at {save_path}")
            continue

        prompt = f"make them look {emotion}, express {emotion}"
        with torch.no_grad():
            result_image = pipeline(
                prompt=prompt,
                image=original_image,
                num_inference_steps=4,
                guidance_scale=1.5,
                image_guidance_scale=1.5,
                output_type="pil"
            ).images[0]
            
        result_image.save(save_path)
        generated_refs[emotion] = save_path
        print(f"    -> Generated ref for '{emotion}' at {save_path}")
        
    memory_manager._offload_diffusion()
    return {"reference_images": generated_refs}
