import os
import shutil
<<<<<<< HEAD
import csv
import time
import sys
=======
import traceback
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
from langgraph.graph import StateGraph, END
from core.state import MERMAIDState, create_initial_state
from core.memory_manager import MemoryOrchestrator
from dataloader.emotion6 import load_emotion6

# Import Agents
from agents.caption_agent import run_caption_agent
from agents.decision_agent import run_decision_agent
from agents.text_reflection import run_text_reflection
from agents.augmentation import run_augmentation_agent
from agents.visual_reflection import run_visual_reflection

def build_graph(memory_manager: MemoryOrchestrator) -> StateGraph:
    """
    Builds and compiles the LangGraph workflow matching Algorithm 1 of the paper.
    """
    # --- Node wrappers ---
    def caption_node(state: MERMAIDState):
        return run_caption_agent(state, memory_manager)

    def decision_node(state: MERMAIDState):
        return {**run_decision_agent(state, memory_manager), "query_mode": "qinit"}

    def decision_after_text_node(state: MERMAIDState):
        result = run_decision_agent(state, memory_manager)
        result["query_mode"] = "qtext"
        if state.get("query_mode") == "qvis":
            result["iteration"] = state.get("iteration", 0) + 1
        return result

    def decision_after_visual_node(state: MERMAIDState):
        return {**run_decision_agent(state, memory_manager), "query_mode": "qvis"}

    def text_reflection_node(state: MERMAIDState):
        return run_text_reflection(state, memory_manager)

    def augmentation_node(state: MERMAIDState):
        return run_augmentation_agent(state, memory_manager)

    def visual_reflection_node(state: MERMAIDState):
        return run_visual_reflection(state, memory_manager)

<<<<<<< HEAD
    def cleanup_node(state: MERMAIDState):
        # Cache is maintained across loops and cleared per-image in the outer evaluation loop
        return {}
=======
    # --- Graph assembly ------------------------------------------------------
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f

    # --- Graph assembly ---
    workflow = StateGraph(MERMAIDState)

    workflow.add_node("captioning",             caption_node)
<<<<<<< HEAD
    workflow.add_node("decision",               decision_node)
    workflow.add_node("text_reflection",        text_reflection_node)
    workflow.add_node("decision_after_text",    decision_after_text_node)
    workflow.add_node("augmentation",           augmentation_node)
    workflow.add_node("visual_reflection",      visual_reflection_node)
    workflow.add_node("decision_after_visual",  decision_after_visual_node)
    workflow.add_node("cleanup",                cleanup_node)
=======
    workflow.add_node("decision",               decision_node)              # Block 1
    workflow.add_node("text_reflection",        text_reflection_node)       # Block 2 / loop
    workflow.add_node("decision_after_text",    decision_after_text_node)   # Block 2 end / loop
    workflow.add_node("augmentation",           augmentation_node)          # Block 3
    workflow.add_node("visual_reflection",      visual_reflection_node)     # Block 3
    workflow.add_node("decision_after_visual",  decision_after_visual_node) # Block 3 inner
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f

    # --- Routing ---
    workflow.set_entry_point("captioning")
    workflow.add_edge("captioning", "decision")
    workflow.add_edge("decision", "text_reflection")

    def route_after_text(state: MERMAIDState) -> str:
        if state.get("textual_feedback", {}).get("status") == "good_enough":
            return END
        return "decision_after_text"

    workflow.add_conditional_edges(
        "text_reflection",
        route_after_text,
        {"cleanup": "cleanup", "decision_after_text": "decision_after_text"}
    )

    workflow.add_edge("decision_after_text", "augmentation")
    workflow.add_edge("augmentation", "visual_reflection")

    def route_after_visual(state: MERMAIDState) -> str:
        if (
            state.get("visual_feedback", {}).get("status") == "good_enough"
            or state["iteration"] >= state["max_iterations"]
        ):
            return END
        return "decision_after_visual"

<<<<<<< HEAD
    workflow.add_conditional_edges(
        "visual_reflection",
        route_after_visual,
        {"cleanup": "cleanup", "decision_after_visual": "decision_after_visual"}
    )

    workflow.add_edge("decision_after_visual", "text_reflection")
    workflow.add_edge("cleanup", END)
=======
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
          good_enough   → END
          needs_revision → decision_after_text (which then goes to augmentation)

        Iteration is incremented here only when we're re-entering the loop.
        """
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
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f

    return workflow.compile()


if __name__ == "__main__":
<<<<<<< HEAD
    print("=== Initializing MERMAID with LLaVA-NeXT-13B (4-bit NF4) ===")
    
    # 1. Initialize Memory Manager
    mem_manager = MemoryOrchestrator(mllm_model_id="llava-hf/llava-v1.6-vicuna-13b-hf")
=======
    import time
    from core.state import create_initial_state
    from dataloader.emoset import load_emoset

    print("=== Initializing Compute-Efficient MERMAID ===")
    
    # 1. Initialize Memory Orchestrator with LLaVA 1.5 7B
    mem_manager = MemoryOrchestrator(mllm_model_id="llava-hf/llava-1.5-7b-hf")
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f

    # 2. Compile LangGraph workflow
    app = build_graph(mem_manager)

<<<<<<< HEAD
    # 3. Load Emotion6 Dataset (1000 images balanced across classes)
    dataset_search_path = "/kaggle/input/emotion6" if os.path.exists("/kaggle/input/emotion6") else "/kaggle/input"
    if not os.path.exists(dataset_search_path):
        dataset_search_path = "./emotion6"

    print(f"\n=== Loading Emotion6 Dataset from '{dataset_search_path}' ===")
    dataset = load_emotion6(dataset_search_path, total_limit=1000)

    if not dataset:
        print(f"[Error] No Emotion6 images found in {dataset_search_path}. Exiting.")
        sys.exit(1)

    candidate_emotions = sorted(list({item["ground_truth"] for item in dataset}))
    print(f"Candidate Emotion Classes ({len(candidate_emotions)}): {candidate_emotions}")

    total = len(dataset)
=======
    # 3. Load EmoSet Dataset
    print(f"\n=== Loading EmoSet Dataset (1000 images) ===")
    dataset = load_emoset("data/emoset_1000", total_limit=1000)
    
    if not dataset:
        print("Dataset not found or empty. Exiting.")
        exit(1)
        
    subset = dataset[10:20]
    
    # EmoSet 8 classes
    candidate_emotions = [
        "Amusement", "Anger", "Awe", "Contentment",
        "Disgust", "Excitement", "Fear", "Sadness"
    ]
    
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
    correct_predictions = 0

    os.makedirs("results", exist_ok=True)
<<<<<<< HEAD
    csv_filename = "results/emotion6_evaluation_llavanext_13b.csv"
=======
    csv_filename = "results/qformer_emoset_evaluation_llava_1.5_7b.csv"
    
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
    print(f"\n=== Starting Evaluation on {total} images (Logging to {csv_filename}) ===")
    start_time = time.time()

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_path", "predicted_emotion", "ground_truth", "is_correct"])
        csv_file.flush()

        for i, item in enumerate(dataset):
            image_path = item["image_path"]
            ground_truth = item["ground_truth"]
            img_stem = os.path.splitext(os.path.basename(image_path))[0]

            # Clear cache from previous image to keep disk usage light
            if i > 0 and os.path.exists("generated_refs"):
                shutil.rmtree("generated_refs")

            print(f"\n{'='*70}")
            print(f"=== Testing Image [{i+1}/{total}]: {os.path.basename(image_path)} (GT: {ground_truth}) ===")
            print(f"{'='*70}")

            initial_state = create_initial_state(
                image_path=image_path,
                candidate_emotions=candidate_emotions,
                max_iterations=2,
            )

            t0 = time.time()
            try:
                final_state = app.invoke(initial_state)
                prediction = str(final_state.get("current_prediction") or "")
            except Exception as e:
                print(f"[Error] Graph failed on image {image_path}: {e}")
                traceback.print_exc()
                prediction = "Error"

            elapsed = time.time() - t0
            is_match = prediction.lower() == ground_truth.lower()
            is_correct_flag = 1 if is_match else 0

            csv_writer.writerow([image_path, prediction, ground_truth, is_correct_flag])
            csv_file.flush()
<<<<<<< HEAD

=======
            
            # Clean up reference images AFTER the result is safely written to CSV
            if os.path.exists("generated_refs"):
                shutil.rmtree("generated_refs")
                print(f"--- [Cleanup] Removed generated_refs for image {i+1} ---")
            
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
            if is_match:
                correct_predictions += 1
                print(f"RESULT: [MATCH] Pred: '{prediction}' | GT: '{ground_truth}' ({elapsed:.1f}s)")
            else:
<<<<<<< HEAD
                print(f"RESULT: [MISMATCH] Pred: '{prediction}' | GT: '{ground_truth}' ({elapsed:.1f}s)")

=======
                print(f"RESULT: [FAILED] Predicted: {prediction} | GT: {ground_truth}")

            
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
    end_time = time.time()
    accuracy = (correct_predictions / total) * 100 if total > 0 else 0

    print(f"\n{'='*70}")
    print("=== Evaluation Complete ===")
    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Results CSV: {csv_filename}")
    print(f"{'='*70}")

    sys.stdout.flush()
    os._exit(0)
