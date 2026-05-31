# High-Precision Custom Object Detection Pipeline for Workspace Environments

An end-to-end computer vision pipeline that utilizes transfer learning over a fine-tuned **MobileNetV2** backbone to simultaneously localize and classify multiple targets (`Person`, `Laptop`, `Chair`) within indoor workspace environments.

## Project Overview
Unlike standard image classifiers that only output global tags, this project implements a custom **multi-task dual-headed prediction network**. It features parallel output channels: a classification head predicting categorical probabilities, and a continuous regression head predicting normalized bounding box coordinates $[ymin, xmin, ymax, xmax]$. 

The system utilizes a custom **Masked Absolute Box Loss (L1 Loss)** function to ignore empty padding slots during batch training and optimizes feature maps by unfreezing the top 40 deep structural layers of the backbone network.

## Technologies Used
* **Languages & Frameworks:** Python, TensorFlow, Keras API
* **Data & Annotation Curation:** FiftyOne Vision Framework (Open Images V7 API)
* **Image Processing & Post-Processing:** OpenCV (Open Source Computer Vision Library)
* **Mathematical Operations & Visualizations:** NumPy, Matplotlib

## Dataset Details
* **Source:** Google Open Images V7 Dataset curated dynamically via FiftyOne Zoo.
* **Volume:** 1,500 highly diverse Training Images and 100 Validation Images.
* **Target Classes:** `Person` (Class ID: 0), `Laptop` (Class ID: 1), `Chair` (Class ID: 2).

## Execution Workflow
1. **Hardware Acceleration:** Connects to an active NVIDIA T4 GPU backend.
2. **Data Streaming:** Queries, downloads, and isolates target image subsets.
3. **Preprocessing:** Resizes images to $320 \times 320 \times 3$, normalizes pixels ($0.0$ to $1.0$), and structures annotations into a rigid $10 \times 4$ padded matrix.
4. **Fine-Tuning:** Unblocks the top 40 deep structural layers of the backbone to adjust parameters for localized tracking.
5. **Post-Processing Filter:** Decodes raw coordinates back to full image sizes and runs **Non-Maximum Suppression (NMS)** via OpenCV to eliminate duplicate overlapping frames.

## Training Results Summary
The network achieved stable convergence over its optimization horizon, automatically stopping at **Epoch 30** via Early Stopping to protect against overfitting, and restored the optimal weights from **Epoch 24**.
* **Initial Global Loss (Epoch 6):** 1.1102 (`boxes_loss`: 0.0225, `labels_loss`: 0.6600)
* **Optimal Restored Loss (Epoch 24):** 0.8215 (`boxes_loss`: 0.0177, `labels_loss`: 0.4691)
* **Final Converged Loss (Epoch 30):** 0.7507 (`boxes_loss`: 0.0164, `labels_loss`: 0.4220)

## How to Run the Project
1. Open Google Colab or your local Jupyter Notebook environment.
2. Upload the `Object_Detection_Project.ipynb` file to your workspace.
3. Upload your test images directly to your main workspace directory so the prediction function can find it.
4. Execute the cells sequentially from top to bottom (`Shift + Enter`). 
5. The notebook will automatically handle the package installations and download the required 1,500 data samples in the first few cells before training the model and outputting the visualization grid.