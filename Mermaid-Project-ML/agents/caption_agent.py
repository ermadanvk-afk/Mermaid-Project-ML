import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator

def run_caption_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Calls the MLLM to generate a rich textual description of the input image.
    
    Input:
        - state: The current LangGraph state containing 'image_path'.
        - memory_manager: Memory manager instance that provides access to the loaded LLaVA-NeXT 7B model (4-bit quantized) and processor.
        
    Output:
        - dict: A dictionary with the key 'caption' containing the generated text description to update the state.
        
    What it does:
        1. Instructs the MemoryOrchestrator to ensure the MLLM is loaded into the GPU.
        2. Loads the image using PIL.
        3. Constructs a prompt compatible with LLaVA-NeXT.
        4. Prepares the inputs using the LLaVA processor.
        5. Generates the caption using the model and returns the decoded string.
    """
    print(f"--- [Agent: Captioning] Analyzing {state['image_path']} ---")
    
    # 1. Ensure MLLM is on GPU
    memory_manager.load_mllm()
    
    # 2. Extract model and processor from memory manager
    model = memory_manager.mllm
    processor = memory_manager.processor
    
    # 3. Load Image
    image = Image.open(state['image_path']).convert("RGB")
    
    # 4. Construct prompt for LLaVA-NeXT
    # Assumes standard Mistral/LLaVA-NeXT chat template
    prompt = "[INST] <image>\nDescribe this image in terms of emotional content. Include expressions, body language, scene context, dominant colours, and any object that influences mood. [/INST]"
    
    # 5. Prepare inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # 6. Generate Output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=150)
        
    # Trim the input prompt tokens from the generated output
    input_len = inputs["input_ids"].shape[1]
    caption = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    
    print(f"    -> Generated Caption: '{caption}'")
    
    return {"caption": caption}
