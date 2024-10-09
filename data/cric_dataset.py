import cv2
import json
import numpy as np
import os
import polars as pl
from typing import List, Dict

class CricDataset:
  def __init__(self, dataset_path: str, images_path: str, segmented_cells_path: str) -> None:
    self.classifications = self.deserialize(dataset_path)
    self.images = self.load_images(images_path)
    self.segmented_cells_path = segmented_cells_path
    
    self.classification_map = {
      'ASC-H': 'asch',
      'ASC-US': 'ascus',
      'HSIL': 'hsil',
      'LSIL': 'lsil',
      'SCC': 'scc',
      'Negative for intraepithelial lesion': 'normal'
    }
  
  def deserialize(self, dataset_path: str) -> pl.DataFrame:
    return pl.read_csv(dataset_path)

  def load_images(self,  images_path: str) -> Dict[str, np.ndarray]:
    print(f"[INFO] Loading images from {images_path}...")
    
    images = {}
    
    images_list = self.classifications.select(pl.col("image_filename")).unique()
    
    for image_name in images_list.iter_rows():
      image_path = os.path.join(images_path, image_name[0])
      image = cv2.imread(image_path)
      images[image_name[0]] = image
      
    print(f"[INFO] Total of images loaded: {len(images)}")
    
    return images
  
  def extract_cells(self) -> None:
    print("[INFO] Extracting cells...")
    
    if not os.path.exists(self.segmented_cells_path):
      os.makedirs(self.segmented_cells_path)
    
    failed_count = 0
    
    for row in self.classifications.iter_rows(named=True):
      image_name = row['image_filename']
      cell_id = row['cell_id']
      x = row['nucleus_x']
      y = row['nucleus_y']
      bethesda_system = self.classification_map[row['bethesda_system']]

      image = self.images[image_name]
      cell = self.extract_cell(image, x, y)
      
      classification_path = os.path.join(self.segmented_cells_path, bethesda_system)
      
      if not os.path.exists(classification_path):
        os.makedirs(classification_path)
      
      #save the cell image
      cell_path = os.path.join(classification_path, f"{cell_id}_{image_name}")    
      
      #save image if nparray is valid
      if np.any(cell):
        cv2.imwrite(cell_path, cell)
      else:
        failed_count += 1
        print(f"[WARNING] Cell {cell_id} from image {image_name} is empty!")
            
    print(f"[INFO] Cells extracted successfully! ({failed_count} failed)")
  
  def extract_cell(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
    cell = image[y-50:y+50, x-50:x+50]
    return cell