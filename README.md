<h1 align="center">TreeTracer</h1>
The code automates tree segmentation from geospatial images using Detectron2’s Mask R-CNN. It downloads required data, processes TIFF images, performs instance segmentation, and generates visual outputs with default and custom visualizations, highlighting trees in yellow and backgrounds in purple, saving results for environmental analysis.

## Execution Guide:
1. Clone the repository:
   ```
   https://github.com/kr1shnasomani/TreeTracer.git
   cd TreeTracer
   ```
   
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

3. Enter the path of the data folder (`TreeTracer`) and input image

4. Upon running the code it saves a model, tiff and the result images in their respective folders

## Result:

   Input Image:

   ![image](https://github.com/user-attachments/assets/31c2d91c-65b5-4dae-8d02-29e539d55743)

   Output Image:

   a. `segmentation.jpg`

   ![image](https://github.com/user-attachments/assets/247f6391-ec52-4481-8e00-a2bbd8025d77)

   b. `mask.jpg`

   ![image](https://github.com/user-attachments/assets/8304c570-e40d-4849-b3f3-49be46d733c7)
