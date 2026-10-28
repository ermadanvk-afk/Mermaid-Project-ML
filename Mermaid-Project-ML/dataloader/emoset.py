import os
import glob


def load_emoset(dataset_root: str, total_limit: int = None) -> list[dict]:
    """
    Loads the EmoSet dataset uniformly across classes.
    Supports:
      1. Flat directory: dataset_root/emotion_xxxxx.jpg
      2. Nested directories: dataset_root/emotion_class/*.jpg
    
    Args:
        dataset_root: Path to the root folder of EmoSet.
        total_limit: If specified, returns exactly this many images, distributed
                     as evenly as possible across all available classes.
        
    Returns:
        List of dictionaries containing 'image_path' and 'ground_truth' label.
    """
    if not os.path.exists(dataset_root):
        print(f"Warning: EmoSet directory '{dataset_root}' not found.")
        return []

    grouped_data = {}

    # Check for flat directory structure: dataset_root/*.jpg
    flat_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        flat_files.extend(glob.glob(os.path.join(dataset_root, ext)))

    if flat_files:
        for img_path in sorted(flat_files):
            filename = os.path.basename(img_path)
            emotion = filename.split('_')[0].capitalize()
            grouped_data.setdefault(emotion, []).append({
                "image_path": img_path,
                "ground_truth": emotion,
            })
    else:
        # Nested directory structure
        for emotion_dir in sorted(os.listdir(dataset_root)):
            dir_path = os.path.join(dataset_root, emotion_dir)
            if os.path.isdir(dir_path):
                emotion = emotion_dir.capitalize()
                grouped_data[emotion] = []
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for img_path in sorted(glob.glob(os.path.join(dir_path, ext))):
                        grouped_data[emotion].append({
                            "image_path": img_path,
                            "ground_truth": emotion,
                        })

    if not grouped_data:
        return []

    # If no limit, flatten and return
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
    for i, emotion in enumerate(sorted(grouped_data.keys())):
        quota = base_quota + (1 if i < remainder else 0)
        class_images = grouped_data[emotion]
        take_count = min(quota, len(class_images))
        final_data.extend(class_images[:take_count])

    return final_data
