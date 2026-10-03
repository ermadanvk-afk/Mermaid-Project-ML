# Low-Level Design (LLD): Compute-Efficient MERMAID Framework

## 1. Introduction
This document details the Low-Level Design for the compute-efficient MERMAID framework, focusing on the LangGraph state schema, node implementations, memory management, and data structures.

## 2. State Management (LangGraph Schema)
The LangGraph state dictionary (`MERMAIDState`) holds all context for a single inference request.

```python
from typing import TypedDict, List, Dict, Optional

class MERMAIDState(TypedDict):
    # Inputs
    image_path: str
    candidate_emotions: List[str]
    
    # Pre-computation
    caption: str
    
    # Iteration Tracking
    iteration: int
    max_iterations: int
    
    # Current State
    current_prediction: Optional[str]
    
    # Feedback Data
    textual_feedback: Optional[Dict] # {status, suggestion, justification}
    visual_feedback: Optional[Dict]  # {status, suggestion, justification}
    
    # Generated Assets
    reference_images: Dict[str, str] # {emotion_label: generated_image_path}
```

## 3. Node Definitions (Agents)

### 3.1 Node: `CaptioningNode`
*   **Action:** Calls the MLLM to describe the image.
*   **Input State:** `image_path`
*   **Output State:** Updates `caption`
*   **Model Constraints:** Requires MLLM loaded in VRAM.

### 3.2 Node: `DecisionNode`
*   **Action:** Prompts the MLLM for the dominant emotion. The prompt dynamically includes `textual_feedback` and `visual_feedback` if `iteration > 0`.
*   **Input State:** `image_path`, `caption`, `candidate_emotions`, `textual_feedback`, `visual_feedback`
*   **Output State:** Updates `current_prediction`
*   **Structured Output:** Single string (must be in `candidate_emotions`).

### 3.3 Node: `TextReflectionNode`
*   **Action:** Prompts the MLLM (Text-only) to evaluate `current_prediction` against `caption`.
*   **Input State:** `caption`, `current_prediction`, `candidate_emotions`
*   **Output State:** Updates `textual_feedback`
*   **Structured Output (JSON schema):**
    ```json
    {
      "status": "good_enough" | "needs_revision",
      "suggestion": "string",
      "justification": "string"
    }
    ```

### 3.4 Node: `AugmentationNode`
*   **Action:** Iterates through `candidate_emotions` and generates an image for each using the diffusion model + LCM.
*   **Input State:** `image_path`, `candidate_emotions`
*   **Output State:** Updates `reference_images`
*   **Model Constraints:** Requires Model Manager to swap out MLLM and load Diffusion model to VRAM.

### 3.5 Node: `VisualReflectionNode`
*   **Action:** Prompts the MLLM to visually compare the original image with the generated `reference_images` in the context of the `current_prediction`.
*   **Input State:** `image_path`, `reference_images`, `current_prediction`, `candidate_emotions`
*   **Output State:** Updates `visual_feedback`
*   **Model Constraints:** Requires Model Manager to swap out Diffusion model and load MLLM to VRAM.

## 4. Sequential Memory Manager (`MemoryOrchestrator`)
A class to handle GPU VRAM gracefully, ensuring models don't crash the 12GB limit.

```python
class MemoryOrchestrator:
    def __init__(self):
        self.mllm_loaded = False
        self.diffusion_loaded = False
        # Initialize models on CPU RAM
        
    def load_mllm(self):
        if self.diffusion_loaded:
            self._offload_diffusion()
        if not self.mllm_loaded:
            # Move MLLM to 'cuda'
            self.mllm_loaded = True
            
    def load_diffusion(self):
        if self.mllm_loaded:
            self._offload_mllm()
        if not self.diffusion_loaded:
            # Move Diffusion to 'cuda'
            self.diffusion_loaded = True
            
    def _offload_mllm(self):
        # Move MLLM to 'cpu'
        # Execute torch.cuda.empty_cache()
        self.mllm_loaded = False
        
    def _offload_diffusion(self):
        # Move Diffusion to 'cpu'
        # Execute torch.cuda.empty_cache()
        self.diffusion_loaded = False
```

## 5. LangGraph Edge Logic
*   **`route_after_text_reflection(state)`**:
    *   If `state['textual_feedback']['status'] == 'good_enough'`, transition to `END`.
    *   Else, transition to `AugmentationNode`.
*   **`route_after_visual_reflection(state)`**:
    *   If `state['visual_feedback']['status'] == 'good_enough'` OR `state['iteration'] >= state['max_iterations']`, transition to `END`.
    *   Else, increment `iteration` and transition to `DecisionNode`.
