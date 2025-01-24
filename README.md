<h1 align="center">TreeTracer</h1>
The code automates tree segmentation from geospatial images using Detectron2’s Mask R-CNN. It downloads required data, processes TIFF images, performs instance segmentation, and generates visual outputs with default and custom visualizations, highlighting trees in yellow and backgrounds in purple, saving results for environmental analysis.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   pip install opencv-python-headless torch torchvision detectron2 rioxarray intake numpy pillow
   ```

2. Enter the path of the data folder (`TreeTracer`) and input image

3. Upon running the code it saves a model, tiff and the result images in their respective folders

## Result::

   Input Image:

   ![image](https://github.com/user-attachments/assets/31c2d91c-65b5-4dae-8d02-29e539d55743)

   Output Image:

   a. `segmentation.jpg`

   ![image](https://github.com/user-attachments/assets/247f6391-ec52-4481-8e00-a2bbd8025d77)

   b. `mask.jpg`

   ![image](https://github.com/user-attachments/assets/8304c570-e40d-4849-b3f3-49be46d733c7)

## Overview:
This code implements a tree segmentation and visualization pipeline using Detectron2 and geospatial data tools. Here's a breakdown:

#### **1. Setup and Configuration:**
- **Data Directories**: Prepares directories for input TIFF files, model files, and output visualizations.
- **URLs and Downloads**: Automates the download of:
  - Geospatial image (TIFF).
  - Pre-trained Mask R-CNN model file.
- **Catalog YAML**: Creates a metadata catalog for the TIFF file using Intake.

#### **2. Geospatial Image Handling:**
- Loads the geospatial raster image (TIFF) using `rioxarray`.
- Integrates Intake for catalog management and metadata association.

#### **3. Detectron2 Model Configuration:**
- **Model Setup**:
  - Loads Detectron2's Mask R-CNN (COCO pre-trained).
  - Configures for CPU-based inference.
  - Sets thresholds and number of classes (single class: "tree").
- **Model Weights**: Verifies and loads the pre-trained model.

#### **4. Image Segmentation:**
- **Input Image**: Reads the image to segment using OpenCV.
- **Instance Segmentation**:
  - Runs inference using the Detectron2 predictor.
  - Outputs instance segmentation results.

#### **5. Visualization:**
- **Default Visualization**:
  - Uses Detectron2's `Visualizer` for standard instance segmentation output.
  - Saves as `segmentation.jpg`.
- **Custom Visualization**:
  - Applies yellow color for trees and purple for non-tree regions.
  - Saves as `mask.jpg`.

#### **6. Output:** 
Final visualizations (`segmentation.jpg` and `mask.jpg`) are saved in the `output` directory.

This pipeline integrates geospatial data handling, pre-trained instance segmentation models, and custom visualization techniques for environmental analysis tasks.
