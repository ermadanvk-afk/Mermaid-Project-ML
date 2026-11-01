import os
import shutil
import traceback
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
    # --- Node wrappers -------------------------------------------------------

    def caption_node(state: MERMAIDState):
        return run_caption_agent(state, memory_manager)

    def decision_node(state: MERMAIDState):
        """Block 1: initial prediction with Qinit (no feedback)."""
        state_for_agent = {**state, "query_mode": "qinit"}
        result = run_decision_agent(state_for_agent, memory_manager)
        result["query_mode"] = "qinit"
        return result

    def decision_after_text_node(state: MERMAIDState):
        iteration = state.get("iteration", 0)
        if state.get("query_mode") == "qvis":
            iteration += 1

        state_for_agent = {**state, "query_mode": "qtext", "iteration": iteration}
        result = run_decision_agent(state_for_agent, memory_manager)
        result["query_mode"] = "qtext"
        result["iteration"] = iteration
        return result

    def decision_after_visual_node(state: MERMAIDState):
        """Block 3: re-predict with Qvis (visual feedback injected)."""
        state_for_agent = {**state, "query_mode": "qvis"}
        result = run_decision_agent(state_for_agent, memory_manager)
        result["query_mode"] = "qvis"
        return result

    def text_reflection_node(state: MERMAIDState):
        return run_text_reflection(state, memory_manager)

    def augmentation_node(state: MERMAIDState):
        return run_augmentation_agent(state, memory_manager)

    def visual_reflection_node(state: MERMAIDState):
        return run_visual_reflection(state, memory_manager)

    # --- Graph assembly ------------------------------------------------------

    workflow = StateGraph(MERMAIDState)

    workflow.add_node("captioning",             caption_node)
    workflow.add_node("decision",               decision_node)              # Block 1
    workflow.add_node("text_reflection",        text_reflection_node)       # Block 2 / loop
    workflow.add_node("decision_after_text",    decision_after_text_node)   # Block 2 end / loop
    workflow.add_node("augmentation",           augmentation_node)          # Block 3
    workflow.add_node("visual_reflection",      visual_reflection_node)     # Block 3
    workflow.add_node("decision_after_visual",  decision_after_visual_node) # Block 3 inner

    # --- Routing functions ---------------------------------------------------

    def route_after_text_reflection(state: MERMAIDState) -> str:
        if state.get("textual_feedback", {}).get("status") == "good_enough":
            return END
        return "decision_after_text"

    def route_after_visual_reflection(state: MERMAIDState) -> str:
        if (
            state.get("visual_feedback", {}).get("status") == "good_enough"
            or state["iteration"] >= state["max_iterations"]
        ):
            return END
        return "decision_after_visual"

    def route_after_decision_after_visual(state: MERMAIDState) -> str:
        # Always go to text_reflection to cross-verify (Algorithm 1, Line 15)
        return "text_reflection"

    def route_after_text_reflection_in_loop(state: MERMAIDState) -> str:
        if (
            state.get("textual_feedback", {}).get("status") == "good_enough"
            or state.get("iteration", 0) >= state.get("max_iterations", 3)
        ):
            return END
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
            END: END,
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
            END: END,
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

    return workflow.compile()


if __name__ == "__main__":
    import time
    from core.state import create_initial_state
    from dataloader.lai_gai import load_lai_gai

    print("=== Initializing Compute-Efficient MERMAID ===")
    
    # 1. Initialize Memory Orchestrator with LLaVA 1.5 7B (as used in the paper)
    mem_manager = MemoryOrchestrator(mllm_model_id="llava-hf/llava-1.5-7b-hf")

    # 2. Build Graph
    app = build_graph(mem_manager)

    # 3. Load LAI-GAI Dataset
    print(f"\n=== Loading LAI-GAI Dataset ===")
    dataset = load_lai_gai("data/lai_gai_labeled_images")
    
    if not dataset:
        print("Dataset not found or empty. Exiting.")
        exit(1)
        
    subset = dataset
    
    # LAI-GAI 12 classes
    candidate_emotions = sorted(list(set(item["ground_truth"] for item in dataset)))
    
    correct_predictions = 0
    total = len(subset)
    
    import csv
    import os
    
    # Ensure a results directory exists
    os.makedirs("results", exist_ok=True)
    csv_filename = "results/laigai_evaluation_llava_1.5_7b.csv"
    
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
                prediction = str(final_state.get("current_prediction") or "")
            except Exception as e:
                print(f"[Error] Graph failed on image {image_path}: {e}")
                traceback.print_exc()
                prediction = "Error"
                
            is_match = prediction.lower() == ground_truth.lower()
            is_correct_flag = 1 if is_match else 0
            
            # Write to CSV
            csv_writer.writerow([image_path, prediction, ground_truth, is_correct_flag])
            # Flush immediately so we don't lose data if it crashes
            csv_file.flush()
            
            # Clean up reference images AFTER the result is safely written to CSV
            if os.path.exists("generated_refs"):
                shutil.rmtree("generated_refs")
                print(f"--- [Cleanup] Removed generated_refs for image {i+1} ---")
            
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

    # Force flush and exit to prevent huggingface/pytorch background threads from hanging
    import sys
    sys.stdout.flush()
    os._exit(0)

