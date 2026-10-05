import os
import glob

def load_emoset(dataset_root: str) -> list[dict]:
    """
    Loads the EmoSet dataset.
    Assumes standard directory structure: dataset_root/emotion_class/image.jpg
    
    Args:
        dataset_root: Path to the root folder of EmoSet.
        
    Returns:
        List of dictionaries containing 'image_path' and 'ground_truth' label.
    """
    data = []
    if not os.path.exists(dataset_root):
        print(f"Warning: EmoSet directory '{dataset_root}' not found.")
        return data
        
    # EmoSet is stored as flat files: dataset_root/emotion_xxxxx.jpg
    for img_path in glob.glob(os.path.join(dataset_root, "*.jpg")):
        filename = os.path.basename(img_path)
        # Parse emotion from filename (e.g., amusement_00290.jpg -> amusement)
        emotion = filename.split('_')[0].capitalize()
        data.append({
            "image_path": img_path,
            "ground_truth": emotion
        })
        
    return data
