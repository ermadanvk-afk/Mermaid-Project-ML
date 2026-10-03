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
