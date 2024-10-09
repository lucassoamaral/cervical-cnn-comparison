from data.cric_dataset import CricDataset

cd = CricDataset('data/classifications.csv', 'data/full_images', 'data/segmented_images')
cd.extract_cells()