import os
import shutil
from langgraph.graph import StateGraph, END
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator

# Import Agents
from agents.caption_agent import run_caption_agent
from agents.decision_agent import run_decision_agent
from agents.text_reflection import run_text_reflection
from agents.augmentation import run_augmentation_agent
from agents.visual_reflection import run_visual_reflection

def build_graph(memory_manager: MemoryOrchestrator) -> StateGraph:
    """
    Builds and compiles the LangGraph workflow matching Algorithm 1 of the paper.

    Graph structure (3 blocks):

    BLOCK 1 — Initial pass:
        captioning → decision[Qinit] → text_reflection

    BLOCK 2 — Textual guidance:
        text_reflection:
            good_enough  → cleanup → END
            needs_revision → decision_after_text[Qtext]
        decision_after_text → augmentation

    BLOCK 3 — Visual loop (repeats up to max_iterations):
        augmentation → visual_reflection
        visual_reflection:
            good_enough  → cleanup → END
            needs_revision → decision_after_visual[Qvis]
        decision_after_visual → text_reflection
        text_reflection (re-run):
            good_enough  → cleanup → END
            needs_revision → decision_after_text[Qtext] → augmentation (new iteration)
    """

    # --- Node wrappers -------------------------------------------------------

    def caption_node(state: MERMAIDState):
        return run_caption_agent(state, memory_manager)

    def decision_node(state: MERMAIDState):
        """Block 1: initial prediction with Qinit (no feedback)."""
        return {**run_decision_agent(state, memory_manager), "query_mode": "qinit"}

    def decision_after_text_node(state: MERMAIDState):
        """
        Block 2 / loop re-entry: re-predict with Qtext (textual feedback injected).
        Increments iteration counter when coming from the visual loop (query_mode == 'qvis'),
        matching the for-loop counter in Algorithm 1, Line 8.
        """
        result = run_decision_agent(state, memory_manager)
        result["query_mode"] = "qtext"
        # Increment iteration only when looping back (not on the first Block 2 pass)
        if state.get("query_mode") == "qvis":
            result["iteration"] = state.get("iteration", 0) + 1
        return result

    def decision_after_visual_node(state: MERMAIDState):
        """Block 3: re-predict with Qvis (visual feedback injected)."""
        return {**run_decision_agent(state, memory_manager), "query_mode": "qvis"}

    def text_reflection_node(state: MERMAIDState):
        return run_text_reflection(state, memory_manager)

    def augmentation_node(state: MERMAIDState):
        return run_augmentation_agent(state, memory_manager)

    def visual_reflection_node(state: MERMAIDState):
        return run_visual_reflection(state, memory_manager)

    def cleanup_node(state: MERMAIDState):
        """Removes temporary reference images generated during augmentation."""
        print("--- [Cleanup] Removing temporary reference images ---")
        if os.path.exists("generated_refs"):
            shutil.rmtree("generated_refs")
        return {"reference_images": {}}

    # --- Graph assembly ------------------------------------------------------

    workflow = StateGraph(MERMAIDState)

    workflow.add_node("captioning",             caption_node)
    workflow.add_node("decision",               decision_node)              # Block 1
    workflow.add_node("text_reflection",        text_reflection_node)       # Block 2 / loop
    workflow.add_node("decision_after_text",    decision_after_text_node)   # Block 2 end / loop
    workflow.add_node("augmentation",           augmentation_node)          # Block 3
    workflow.add_node("visual_reflection",      visual_reflection_node)     # Block 3
    workflow.add_node("decision_after_visual",  decision_after_visual_node) # Block 3 inner
    workflow.add_node("cleanup",                cleanup_node)

    # --- Routing functions ---------------------------------------------------

    def route_after_text_reflection(state: MERMAIDState) -> str:
        """
        After text reflection:
          good_enough   → accept current prediction, exit.
          needs_revision → call Decision Agent again with Qtext guidance.
        """
        if state.get("textual_feedback", {}).get("status") == "good_enough":
            return "cleanup"
        return "decision_after_text"

    def route_after_visual_reflection(state: MERMAIDState) -> str:
        """
        After visual reflection:
          good_enough or max_iterations reached → accept, exit.
          needs_revision → call Decision Agent with Qvis guidance.
        """
        if (
            state.get("visual_feedback", {}).get("status") == "good_enough"
            or state["iteration"] >= state["max_iterations"]
        ):
            return "cleanup"
        return "decision_after_visual"

    def route_after_decision_after_visual(state: MERMAIDState) -> str:
        """
        After Decision[Qvis], run Text Reflection again (Algorithm 1, Line 15).
        Then:
          good_enough   → exit.
          needs_revision → increment iteration, Decision[Qtext] → Augmentation (new loop).
        The text_reflection node will be reused; its routing handles the branching.
        """
        # Always go to text_reflection to cross-verify (Algorithm 1, Line 15)
        return "text_reflection"

    def route_after_text_reflection_in_loop(state: MERMAIDState) -> str:
        """
        Text reflection can be reached from two paths:
          1. After initial decision (Block 2) — query_mode == "qinit"
          2. After Decision[Qvis]            — query_mode == "qvis"

        In both cases the routing logic is the same:
          good_enough   → cleanup
          needs_revision → decision_after_text (which then goes to augmentation)

        Iteration is incremented here only when we're re-entering the loop.
        """
        if state.get("textual_feedback", {}).get("status") == "good_enough":
            return "cleanup"
        # If we came from the visual loop, increment iteration counter
        # (Algorithm 1: the for-loop counter advances each time we loop back)
        return "decision_after_text"

    # --- Edges ---------------------------------------------------------------

    # Block 1
    workflow.set_entry_point("captioning")
    workflow.add_edge("captioning", "decision")
    workflow.add_edge("decision", "text_reflection")

    # Text reflection — shared router for both Block 2 entry and loop re-entry
    workflow.add_conditional_edges(
        "text_reflection",
        route_after_text_reflection_in_loop,
        {
            "cleanup": "cleanup",
            "decision_after_text": "decision_after_text",
        }
    )

    # Block 2 end → Block 3 start
    workflow.add_edge("decision_after_text", "augmentation")

    # Block 3: augmentation → visual reflection
    workflow.add_edge("augmentation", "visual_reflection")

    # Visual reflection routing
    workflow.add_conditional_edges(
        "visual_reflection",
        route_after_visual_reflection,
        {
            "cleanup": "cleanup",
            "decision_after_visual": "decision_after_visual",
        }
    )

    # Decision[Qvis] → Text Reflection (Algorithm 1, Line 15)
    # Increment iteration before looping back to augmentation
    workflow.add_conditional_edges(
        "decision_after_visual",
        lambda state: "text_reflection",
        {"text_reflection": "text_reflection"}
    )

    # Cleanup always exits
    workflow.add_edge("cleanup", END)

    return workflow.compile()


if __name__ == "__main__":
    import time
    from core.state import create_initial_state
    from dataloader import load_emotion6

    print("=== Initializing Compute-Efficient MERMAID ===")
    
    # 1. Initialize Memory Orchestrator
    mem_manager = MemoryOrchestrator()

    # 2. Build Graph
    app = build_graph(mem_manager)

    # 3. Load Dataset
    print(f"\n=== Loading Emotion6 Dataset ===")
    dataset = load_emotion6("data/Emotion6/images")
    if not dataset:
        print("Dataset not found or empty. Exiting.")
        exit(1)
        
    # Limit to 10 images for testing
    limit = 10
    subset = dataset[:limit]
    
    # Emotion6 classes
    candidate_emotions = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
    
    correct_predictions = 0
    total = len(subset)
    
    import csv
    import os
    
    # Ensure a results directory exists
    os.makedirs("results", exist_ok=True)
    csv_filename = "results/emotion6_evaluation.csv"
    
    print(f"\n=== Starting Evaluation on {total} images (Logging to {csv_filename}) ===")
    start_time = time.time()
    
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(["image_path", "predicted_emotion", "ground_truth", "is_correct"])
        
        for i, item in enumerate(subset):
            image_path = item["image_path"]
            ground_truth = item["ground_truth"]
            
            print(f"\n--- Testing Image {i+1}/{total}: {image_path} ---")
            print(f"Ground Truth: {ground_truth}")
            
            initial_state = create_initial_state(
                image_path=image_path,
                candidate_emotions=candidate_emotions,
                max_iterations=3,
            )
            
            # Run graph
            try:
                final_state = app.invoke(initial_state)
                prediction = final_state.get("current_prediction", "")
            except Exception as e:
                print(f"[Error] Graph failed on image {image_path}: {e}")
                prediction = "Error"
                
            is_match = prediction.lower() == ground_truth.lower()
            is_correct_flag = 1 if is_match else 0
            
            # Write to CSV
            csv_writer.writerow([image_path, prediction, ground_truth, is_correct_flag])
            # Flush immediately so we don't lose data if it crashes
            csv_file.flush()
            
            if is_match:
                correct_predictions += 1
                print(f"RESULT: [SUCCESS] Predicted: {prediction} | GT: {ground_truth}")
            else:
                print(f"RESULT: [FAILED] Predicted: {prediction} | GT: {ground_truth}")
            
    end_time = time.time()
    accuracy = (correct_predictions / total) * 100 if total > 0 else 0
    
    print("\n=== Evaluation Complete ===")
    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Time: {end_time - start_time:.2f} seconds")

