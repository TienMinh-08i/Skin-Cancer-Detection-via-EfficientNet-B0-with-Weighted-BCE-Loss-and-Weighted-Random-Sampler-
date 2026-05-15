"""Generate metadata CSV from image directory."""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def generate_metadata_from_images(image_dir: str, output_csv: str, target_col_name: str = "target") -> None:
    """
    Automatically generate metadata CSV from images in directory.
    
    Args:
        image_dir: Path to directory containing images
        output_csv: Path to save generated CSV
        target_col_name: Column name for target labels (will be filled with dummy 0s)
    """
    image_dir = Path(image_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions])
    
    print(f"Found {len(images)} images")
    
    # Extract image IDs
    data = []
    for img_path in tqdm(images, desc="Processing images"):
        # Extract ID from filename (remove extension)
        image_id = img_path.stem  # e.g., "ISIC_0063233" from "ISIC_0063233.jpg"
        
        data.append({
            'isic_id': image_id,
            'patient_id': f'patient_{len(data)}',  # Dummy patient ID
            target_col_name: 0  # Dummy target label (all benign/0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Generated metadata CSV: {output_csv}")
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nNote: All labels set to 0 (benign). Update manually if needed!")


if __name__ == "__main__":
    # Generate metadata
    image_dir = "/home/22010759/Minhk16/como/content/dataset/image"
    output_csv = "/home/22010759/Minhk16/como/content/dataset/train-metadata-fixed.csv"
    
    generate_metadata_from_images(image_dir, output_csv)
