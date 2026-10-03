import os
import glob

def load_emotion6(dataset_root: str) -> list[dict]:
    """
    Loads the Emotion6 dataset.
    Assumes standard directory structure: dataset_root/emotion_class/image.jpg
    
    Args:
        dataset_root: Path to the root folder of Emotion6.
        
    Returns:
        List of dictionaries containing 'image_path' and 'ground_truth' label.
    """
    data = []
    if not os.path.exists(dataset_root):
        print(f"Warning: Emotion6 directory '{dataset_root}' not found.")
        return data
        
    for emotion_dir in os.listdir(dataset_root):
        dir_path = os.path.join(dataset_root, emotion_dir)
        if os.path.isdir(dir_path):
            for img_ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in glob.glob(os.path.join(dir_path, img_ext)):
                    data.append({
                        "image_path": img_path,
                        "ground_truth": emotion_dir.capitalize()
                    })
    return data
