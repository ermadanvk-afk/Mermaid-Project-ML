# MERMAID Implementation Plan (Blueprint Phase)

## Overview
This plan outlines the steps to build the initial modular blueprint for the compute-efficient MERMAID framework based on our Low-Level Design (LLD). The focus is on strict modularity, strong typing (Pydantic/Typing), and safe memory management.

## Project Structure
We will adopt a highly modular structure to keep agents separated and the codebase clean:

```
mermaid-project/
│
├── requirements.txt
├── main.py                  # Entry point & LangGraph workflow definition
│
├── core/
│   ├── state.py             # LangGraph state (MERMAIDState) definitions
│   ├── memory_manager.py    # MemoryOrchestrator for CPU/GPU model swapping
│   └── schemas.py           # Pydantic schemas for structured LLM outputs
│
└── agents/
    ├── __init__.py
    ├── caption_agent.py     # Image captioning logic
    ├── decision_agent.py    # Initial emotion prediction
    ├── text_reflection.py   # Text-only evaluation
    ├── augmentation.py      # LCM Diffusion image generation
    └── visual_reflection.py # Visual comparison
```

## Phase 1: Environment Setup & Dependencies
We need a robust `requirements.txt` compatible with Python 3.12.
*   **Action:** Create `requirements.txt` with specific library versions where necessary to avoid dependency hell.
*   *Note on CUDA for Windows:* Standard `pip install torch` usually installs the CPU version on Windows. The user will need to install PyTorch with CUDA explicitly using the PyTorch index URL.

## Phase 2: Core Architecture implementation
1.  **`core/state.py`**: Define the `MERMAIDState` using `TypedDict`. This ensures every agent knows exactly what data it is receiving.
2.  **`core/schemas.py`**: Define `Pydantic` models for the reflection agents (e.g., `ReflectionOutput` containing `status`, `suggestion`, `justification`). This is crucial for `instructor` to force structured outputs.
3.  **`core/memory_manager.py`**: Implement the `MemoryOrchestrator` class. Initially, this can just be a "stub" or basic implementation that uses `torch.cuda.empty_cache()` to prove the logic before we wire up the real HuggingFace models.

## Phase 3: Agent Scaffolding (The Blueprint)
For the very first blueprint, we will create the python files for each agent in the `agents/` directory.
Each agent file will have a function with strict type hints for inputs and outputs.
*Example:*
```python
def run_text_reflection(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    # Logic here
    return {"textual_feedback": ...}
```
*Initially, we can mock the actual LLM calls (make them return dummy data) just to test if the LangGraph state routes correctly.*

## Phase 4: LangGraph Orchestration (`main.py`)
1.  Initialize the `StateGraph` using our `MERMAIDState`.
2.  Add all agent functions as nodes to the graph.
3.  Define the conditional edges (the logic that checks `status == "good_enough"` or if `iteration >= max_iterations` to break the loop).
4.  Compile the graph.

## Execution Strategy
1. I will write the `requirements.txt` for you immediately.
2. I will create the directory structure and the core state/memory scripts.
3. I will create the agent blueprints.
4. We will review the blueprint before plugging in the heavy 2B/7B models.



Some Fallback : 
Why the accuracy is 20%:
The fatal flaw right now is that the Textual Reflection Agent is too lenient. Because LLMs tend to be agreeable, when the Text Agent looks at the initial guess (e.g., "Contentment" for a happy scene), the majority of the perspectives just agree and vote good_enough.

Because it votes good_enough on the very first try, the algorithm hits Line 4 and immediately exits, spitting out the wrong prediction. The Augmentation Agent (Image Generation) and Visual Reflection Agent are never being called at all!

How we can fix this and get the accuracy up:
We need to force the system to actually use the Visual Reflection loop. We have two ways to do this:

Option 1: Make the Textual Vote Stricter (Stay true to the paper) Currently, in agents/text_reflection.py, the votes are aggregated by Majority Rule (if 2 out of 3 agents say good_enough, it exits). We can change this to a Unanimous Rule: if even one single perspective votes needs_revision, the system is forced to proceed to Visual Reflection.

Option 2: Remove the Early Exit entirely (Slight deviation from paper) We can modify the LangGraph router in main.py to simply delete the early exit. This guarantees that every single image will undergo at least 1 full round of Emotion-Guided Image Generation and Visual Reflection.

Which option would you like me to implement? (Option 1 is the most scientifically sound, but Option 2 guarantees the visual agents actually get used!).

So why is it failing if the prompts match the paper?
The authors of the paper likely discovered the same issue (that LLMs tend to be "yes-men" and agree with the first guess too easily). To combat this, they mentioned in Section 5.1 that they used an ensemble size of 5 (b=5) and up to 4 distinct perspectives.

Right now, our code only samples 3 perspectives and uses a simple "majority rules" vote (if 2 out of 3 say good_enough, it exits).

Let's Fix It (Option 1)
Since the prompts are already completely faithful to the paper, the best way to solve this is to make the voting system stricter so it doesn't prematurely exit.

Shall I go ahead and modify _aggregate_votes in text_reflection.py to require a Unanimous Vote? (Meaning if even one perspective disagrees, the system will trigger the Image Generation and Visual Reflection loops just like it's supposed to).