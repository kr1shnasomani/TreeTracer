# Import the required libraries
import os
import urllib.request
import cv2
import torch
import numpy as np
import rioxarray
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from PIL import Image
import intake

# Configuration paths
data_folder = r'C:\Users\krish\OneDrive\Desktop\Projects\TreeTracer'
image_path = r'C:\Users\krish\OneDrive\Desktop\image.jpg'

config = {
    'in_geotiff': os.path.join(data_folder, 'tiff'),
    'model': os.path.join(data_folder, 'model'),
    'output': os.path.join(data_folder, 'output')
}

# Create directories if they don't exist
[os.makedirs(val, exist_ok=True) for val in config.values()]

# URLs for downloading
tiff_url = "https://zenodo.org/record/5494629/files/Sep_2014_RGB_602500_646600.tif"
model_url = "https://zenodo.org/record/5515408/files/model_final.pth?download=1"

# Paths to save files
tiff_path = os.path.join(config['in_geotiff'], 'Sep_2014_RGB_602500_646600.tif')
model_path = os.path.join(config['model'], 'model_final.pth')
segmentation_path = os.path.join(config['output'], 'segmentation.jpg')
mask_path = os.path.join(config['output'], 'mask.jpg')

# Download the TIFF file if it doesn't already exist
if not os.path.exists(tiff_path):
    print(f"Downloading TIFF file from {tiff_url} to {tiff_path}")
    urllib.request.urlretrieve(tiff_url, tiff_path)
    print("TIFF file downloaded successfully.")
else:
    print("TIFF file already exists.")

# Download the model file if it doesn't already exist
if not os.path.exists(model_path):
    print(f"Downloading model from {model_url} to {model_path}")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded successfully.")
else:
    print("Model already exists.")

# Write a catalog YAML file for intake
catalog_file = os.path.join(data_folder, 'catalog.yaml')
with open(catalog_file, 'w') as f:
    f.write(f'''
sources:
  sepilok_rgb:
    driver: rasterio
    description: 'NERC RGB images of Sepilok, Sabah, Malaysia (collection)'
    metadata:
      zenodo_doi: "10.5281/zenodo.5494629"
    args:
      urlpath: "{{{{ CATALOG_DIR }}}}/input/tiff/Sep_2014_RGB_602500_646600.tif"
    ''')

# Open the catalog
cat_tc = intake.open_catalog(catalog_file)
print(cat_tc['sepilok_rgb'])

# Open the TIFF file with rioxarray
tc_rgb = rioxarray.open_rasterio(tiff_path)

# Verify model path
print(f"Model saved to: {model_path}")

# Load input image
im = cv2.imread(image_path)

# Setup Detectron2 configuration
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu' 
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.44  

# Load the model
if os.path.exists(model_path):
    cfg.MODEL.WEIGHTS = model_path
    print(f"Model loaded from: {model_path}")
else:
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Initialize the predictor
predictor = DefaultPredictor(cfg)

# Perform inference
outputs = predictor(im)

# Default visualization (Instance segmentation)
v = Visualizer(im[:, :, ::-1], scale=1.5, instance_mode=ColorMode.IMAGE_BW) 
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
default_output_image = cv2.cvtColor(v.get_image()[:, :, :], cv2.COLOR_BGR2RGB)
cv2.imwrite(segmentation_path, cv2.cvtColor(default_output_image, cv2.COLOR_RGB2BGR))
print(f"Default instance segmentation output saved at: {segmentation_path}")

# Custom tree/yellow, background/purple visualization
output_image_custom = np.zeros_like(im, dtype=np.uint8)
instances = outputs["instances"]

if len(instances) > 0:
    masks = instances.pred_masks.to("cpu").numpy()

    # Apply yellow color to tree areas
    for mask in masks:
        output_image_custom[mask] = [0, 255, 255] 

# Apply purpleto non-tree areas
output_image_custom[(output_image_custom == 0).all(axis=2)] = [128, 0, 128]

# Save the custom visualization
cv2.imwrite(mask_path, output_image_custom)
print(f"Custom tree/background output saved at: {mask_path}")