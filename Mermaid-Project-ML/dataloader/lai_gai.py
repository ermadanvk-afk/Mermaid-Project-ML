import os
import re


def load_lai_gai(dataset_root: str, total_limit: int = None) -> list[dict]:
    """
    Loads the LAI-GAI dataset by extracting emotion labels from filenames using regex.
    Expected filename format: <digit>_<emotion>.png (e.g., '000_Craving.png', '013_Nurturant love.png').

    Args:
        dataset_root: Path to the directory containing LAI-GAI images.
        total_limit: Optional limit distributed evenly across emotion classes.

    Returns:
        List of dictionaries with 'image_path' and 'ground_truth'.
    """
    if not os.path.exists(dataset_root):
        print(f"Warning: LAI-GAI directory '{dataset_root}' not found.")
        return []

    # Regex matching: digits followed by underscore and emotion label
    pattern = re.compile(r"^\d+_(.+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
    grouped_data = {}

    for filename in sorted(os.listdir(dataset_root)):
        match = pattern.match(filename)
        if match:
            emotion = match.group(1).strip().capitalize()
            img_path = os.path.join(dataset_root, filename)
            grouped_data.setdefault(emotion, []).append({
                "image_path": img_path,
                "ground_truth": emotion,
            })

    if not grouped_data:
        return []

    # Return all images if no limit is specified
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
        final_data.extend(class_images[: min(quota, len(class_images))])

    return final_data
