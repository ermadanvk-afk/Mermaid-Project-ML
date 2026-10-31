import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator

def run_caption_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Calls LLaVA 1.5 7B (llava-hf/llava-1.5-7b-hf) to generate a rich
    emotional description of the input image via the chat-template pipeline.
    """
    print(f"--- [Agent: Captioning] Analyzing {state['image_path']} with LLaVA 1.5 7B ---")
    
    # 1. Load Image and cap resolution (B7 fix: max dim 512x512)
    image_path = state['image_path']
    raw_image = Image.open(image_path).convert("RGB")
    raw_image.thumbnail((512, 512), Image.LANCZOS)
    
    # 2. Construct prompt as per paper Appendix C
    prompt_text = (
        "Describe this image in terms of emotional content. Include expressions, "
        "body language, scene context, dominant colours, and any object that influences mood."
    )
    
    # 3. Generate caption via LLaVA conversation chat template
    caption = memory_manager.generate_mllm_response(
        image=raw_image,
        prompt_text=prompt_text,
        max_new_tokens=150,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )
    
    print(f"    -> Generated Caption: '{caption}'")
    return {"caption": caption}
