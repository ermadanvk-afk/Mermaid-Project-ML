import os
import glob

def load_emotion6(dataset_root: str, total_limit: int = None) -> list[dict]:
    """
    Loads the Emotion6 dataset uniformly across classes.
    Assumes standard directory structure: dataset_root/emotion_class/image.jpg
    
    Args:
        dataset_root: Path to the root folder of Emotion6.
        total_limit: If specified, returns exactly this many images, distributed
                     as evenly as possible across all available classes.
        
    Returns:
        List of dictionaries containing 'image_path' and 'ground_truth' label.
    """
    if not os.path.exists(dataset_root):
        print(f"Warning: Emotion6 directory '{dataset_root}' not found.")
        return []
        
    # First, collect all images grouped by class
    grouped_data = {}
    for emotion_dir in sorted(os.listdir(dataset_root)):
        dir_path = os.path.join(dataset_root, emotion_dir)
        if os.path.isdir(dir_path):
            grouped_data[emotion_dir] = []
            for img_ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in glob.glob(os.path.join(dir_path, img_ext)):
                    grouped_data[emotion_dir].append({
                        "image_path": img_path,
                        "ground_truth": emotion_dir.capitalize()
                    })
    
    # If no limit, just flatten and return
    if total_limit is None:
        flat_data = []
        for class_list in grouped_data.values():
            flat_data.extend(class_list)
        return flat_data
        
    # Distribute total_limit evenly across classes
    num_classes = len(grouped_data)
    base_quota = total_limit // num_classes
    remainder = total_limit % num_classes
    
    final_data = []
    # Sort keys to ensure deterministic distribution of remainder
    for i, emotion_dir in enumerate(sorted(grouped_data.keys())):
        quota = base_quota + (1 if i < remainder else 0)
        
        # In case a folder has fewer images than the quota (unlikely for Emotion6)
        class_images = grouped_data[emotion_dir]
        take_count = min(quota, len(class_images))
        
        # Take the first 'take_count' images for deterministic uniform sampling
        final_data.extend(class_images[:take_count])
        
    return final_data
