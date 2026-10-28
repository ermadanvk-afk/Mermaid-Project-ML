import os
import glob
import random

def find_emotion6_root(search_root: str = "/kaggle/input") -> str:
    """
    Auto-detects the emotion class directory under search_root.
    Handles structures like:
      /kaggle/input/emotion6/Emotion6/images/<emotion_class>/*.jpg
      or directly:
      /kaggle/input/emotion6/images/<emotion_class>/*.jpg
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, dirs, files in os.walk(search_root):
        sub_image_dirs = []
        for d in sorted(dirs):
            d_path = os.path.join(root, d)
            try:
                if any(f.lower().endswith(valid_exts) for f in os.listdir(d_path)):
                    sub_image_dirs.append(d)
            except (PermissionError, NotADirectoryError):
                continue
        # If folder contains 4+ subdirectories with images, it's the class root
        if len(sub_image_dirs) >= 4:
            return root
    return ""

def load_emotion6(dataset_root: str = "/kaggle/input", total_limit: int = None) -> list[dict]:
    """
    Loads the Emotion6 dataset uniformly across classes.
    Auto-detects nested subdirectories (e.g. emotion6/Emotion6/images) if needed.
    
    Args:
        dataset_root: Root path to Emotion6 or parent search path (e.g. /kaggle/input/emotion6).
        total_limit: If specified, returns exactly this many images, distributed
                     as evenly as possible across all available classes using random sampling.
        
    Returns:
        List of dictionaries containing 'image_path' and 'ground_truth' label.
    """
<<<<<<< HEAD
    resolved_root = dataset_root
    
    # Check if the given root directly contains class folders with images
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    has_class_dirs = False
    if os.path.exists(dataset_root):
        for entry in os.listdir(dataset_root):
            entry_path = os.path.join(dataset_root, entry)
            if os.path.isdir(entry_path):
                for ext in valid_exts:
                    if glob.glob(os.path.join(entry_path, ext)):
                        has_class_dirs = True
                        break
            if has_class_dirs:
                break
                
    if not has_class_dirs:
        # Try finding the class directory inside dataset_root
        auto_found = find_emotion6_root(dataset_root)
        if auto_found:
            resolved_root = auto_found
        else:
            print(f"Warning: Could not find Emotion6 class folders under '{dataset_root}'.")
            return []

    print(f"[Emotion6 Loader] Reading dataset from: {resolved_root}")

    # Collect all images grouped by class
=======
    if not os.path.exists(dataset_root):
        print(f"Warning: Emotion6 directory '{dataset_root}' not found.")
        return []
        
    # If dataset_root points to the outer folder containing 'images/', descend into it
    if os.path.isdir(os.path.join(dataset_root, "images")):
        dataset_root = os.path.join(dataset_root, "images")
        
    # First, collect all images grouped by class
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
    grouped_data = {}
    for emotion_dir in sorted(os.listdir(resolved_root)):
        dir_path = os.path.join(resolved_root, emotion_dir)
        if os.path.isdir(dir_path):
<<<<<<< HEAD
            images = []
            for img_ext in valid_exts:
                for img_path in glob.glob(os.path.join(dir_path, img_ext)):
                    images.append({
=======
            grouped_data[emotion_dir] = []
            for img_ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in sorted(glob.glob(os.path.join(dir_path, img_ext))):
                    grouped_data[emotion_dir].append({
>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
                        "image_path": img_path,
                        "ground_truth": emotion_dir.capitalize()
                    })
            if images:
                grouped_data[emotion_dir] = images
    
    if not grouped_data:
        print(f"Warning: No valid images found under '{resolved_root}'.")
        return []

    print(f"[Emotion6 Loader] Found classes: {list(grouped_data.keys())} ({sum(len(v) for v in grouped_data.values())} total images)")

    # If no limit specified, return all
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
    for i, emotion_dir in enumerate(sorted(grouped_data.keys())):
        quota = base_quota + (1 if i < remainder else 0)
        class_images = grouped_data[emotion_dir]
        
        # Take min if folder has fewer images than quota, sample randomly
        take_count = min(quota, len(class_images))
        sampled = random.sample(class_images, take_count)
        final_data.extend(sampled)

    return final_data
