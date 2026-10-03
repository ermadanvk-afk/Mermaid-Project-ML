import os
import csv

def load_lai_gai(dataset_root: str, annotations_file: str) -> list[dict]:
    """
    Loads the LAI-GAI fairness dataset.
    Because LAI-GAI tracks specific demographic ratings, we assume an 
    annotations CSV file detailing the images and cohort data.
    
    Args:
        dataset_root: Path to the root folder containing the images.
        annotations_file: Path to the CSV containing image-to-cohort metadata.
        
    Returns:
        List of dictionaries containing 'image_path', 'ground_truth', and 'cohort' metadata.
    """
    data = []
    if not os.path.exists(annotations_file):
        print(f"Warning: LAI-GAI annotations file '{annotations_file}' not found.")
        return data
        
    # Assuming standard CSV format: image_filename, emotion_label, demographic_cohort
    with open(annotations_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('image_filename', '')
            img_path = os.path.join(dataset_root, filename)
            
            if os.path.exists(img_path) and filename:
                data.append({
                    "image_path": img_path,
                    "ground_truth": row.get('emotion_label', '').capitalize(),
                    "cohort": row.get('demographic_cohort', 'unknown')
                })
    return data
