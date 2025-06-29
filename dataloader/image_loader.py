import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os

def load_image(csv_path, image_folder):
    matched = []
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        image_name = row['image']
        latex = row['formula']

        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            matched.append({'latex_formula': latex, 'image_path': image_path})

    return matched