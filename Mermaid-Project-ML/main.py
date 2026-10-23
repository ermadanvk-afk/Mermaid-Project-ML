import os
import shutil
import csv
import time
import sys
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

    def cleanup_node(state: MERMAIDState):
        # Cache is maintained across loops and cleared per-image in the outer evaluation loop
        return {}

    # --- Graph assembly ---
    workflow = StateGraph(MERMAIDState)

    workflow.add_node("captioning",             caption_node)
    workflow.add_node("decision",               decision_node)
    workflow.add_node("text_reflection",        text_reflection_node)
    workflow.add_node("decision_after_text",    decision_after_text_node)
    workflow.add_node("augmentation",           augmentation_node)
    workflow.add_node("visual_reflection",      visual_reflection_node)
    workflow.add_node("decision_after_visual",  decision_after_visual_node)
    workflow.add_node("cleanup",                cleanup_node)

    # --- Routing ---
    workflow.set_entry_point("captioning")
    workflow.add_edge("captioning", "decision")
    workflow.add_edge("decision", "text_reflection")

    def route_after_text(state: MERMAIDState) -> str:
        if state.get("textual_feedback", {}).get("status") == "good_enough":
            return "cleanup"
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
            return "cleanup"
        return "decision_after_visual"

    workflow.add_conditional_edges(
        "visual_reflection",
        route_after_visual,
        {"cleanup": "cleanup", "decision_after_visual": "decision_after_visual"}
    )

    workflow.add_edge("decision_after_visual", "text_reflection")
    workflow.add_edge("cleanup", END)

    return workflow.compile()


if __name__ == "__main__":
    print("=== Initializing MERMAID with LLaVA-NeXT-13B (4-bit NF4) ===")
    
    # 1. Initialize Memory Manager
    mem_manager = MemoryOrchestrator(mllm_model_id="llava-hf/llava-v1.6-vicuna-13b-hf")

    # 2. Compile LangGraph workflow
    app = build_graph(mem_manager)

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
    correct_predictions = 0

    os.makedirs("results", exist_ok=True)
    csv_filename = "results/emotion6_evaluation_llavanext_13b.csv"
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
                prediction = final_state.get("current_prediction", "")
            except Exception as e:
                print(f"[Error] Graph failed on image {image_path}: {e}")
                prediction = "Error"

            elapsed = time.time() - t0
            is_match = prediction.lower() == ground_truth.lower()
            is_correct_flag = 1 if is_match else 0

            csv_writer.writerow([image_path, prediction, ground_truth, is_correct_flag])
            csv_file.flush()

            if is_match:
                correct_predictions += 1
                print(f"RESULT: [MATCH] Pred: '{prediction}' | GT: '{ground_truth}' ({elapsed:.1f}s)")
            else:
                print(f"RESULT: [MISMATCH] Pred: '{prediction}' | GT: '{ground_truth}' ({elapsed:.1f}s)")

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
