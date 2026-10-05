import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from qwen_vl_utils import process_vision_info

def run_caption_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Calls the MLLM to generate a rich textual description of the input image.
    """
    print(f"--- [Agent: Captioning] Analyzing {state['image_path']} ---")
    
    # 1. Ensure MLLM is on GPU
    memory_manager.load_mllm()
    
    # 2. Extract model and processor from memory manager
    model = memory_manager.mllm
    processor = memory_manager.processor
    
    # 3. Load Image (Optional: process_vision_info can load it from string path, but PIL also works)
    image_path = state['image_path']
    
    # 4. Construct prompt for Qwen2.5-VL
    prompt_text = "Describe this image in terms of emotional content. Include expressions, body language, scene context, dominant colours, and any object that influences mood."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # 5. Prepare inputs using Qwen processor
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 6. Generate Output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        
    # Trim the input prompt tokens from the generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    
    print(f"    -> Generated Caption: '{caption}'")
    
    return {"caption": caption}
