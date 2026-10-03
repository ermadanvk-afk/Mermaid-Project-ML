import os
import glob

def load_artphoto(dataset_root: str) -> list[dict]:
    """
    Loads the Artphoto dataset.
    Assumes standard directory structure: dataset_root/emotion_class/image.jpg
    
    Args:
        dataset_root: Path to the root folder of Artphoto.
        
    Returns:
        List of dictionaries containing 'image_path' and 'ground_truth' label.
    """
    data = []
    if not os.path.exists(dataset_root):
        print(f"Warning: Artphoto directory '{dataset_root}' not found.")
        return data
        
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path):
            # Original structure: dataset_root/emotion_class/image.jpg
            for img_path in glob.glob(os.path.join(item_path, "*.jpg")):
                data.append({
                    "image_path": img_path,
                    "ground_truth": item.capitalize()
                })
        elif item_path.endswith('.jpg'):
            # Flat structure: dataset_root/emotion_0001.jpg
            emotion = item.split('_')[0].capitalize()
            data.append({
                "image_path": item_path,
                "ground_truth": emotion
            })
    return data
