import os
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator

def run_llava_generation(prompt: str, image_path: str = None, memory_manager: MemoryOrchestrator = None, max_new_tokens: int = 200) -> str:
    """Helper to run LLaVA-NeXT-13B inference with correct Vicuna prompt formatting."""
    memory_manager.load_mllm()
    model = memory_manager.mllm
    processor = memory_manager.processor
    
    if image_path:
        raw_image = Image.open(image_path).convert("RGB")
        formatted_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: <image>\n{prompt} ASSISTANT:"
        )
        inputs = processor(text=formatted_prompt, images=raw_image, return_tensors="pt")
    else:
        formatted_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: {prompt} ASSISTANT:"
        )
        inputs = processor(text=formatted_prompt, return_tensors="pt")
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if "ASSISTANT:" in output_text:
        output_text = output_text.split("ASSISTANT:")[-1]
    return output_text.strip()

def run_caption_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Calls LLaVA-NeXT to generate a rich emotional description of the input image.
    """
    print(f"--- [Agent: Captioning] Analyzing {os.path.basename(state['image_path'])} ---")
    
    prompt = "Describe this image in terms of emotional content. Include expressions, body language, scene context, dominant colours, and any object that influences mood."
    caption = run_llava_generation(prompt, state['image_path'], memory_manager, max_new_tokens=200)
    
    print(f"    -> Generated Caption: '{caption}'")
    return {"caption": caption}
