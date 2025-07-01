Brain Tumor Classification and Segmentation with Explainable AI
This project presents a deep learning pipeline for classifying and segmenting brain tumors from MRI scans, combining performance with interpretability through explainable AI techniques. It features custom modifications to the ResNet-18 architecture for classification, a U-Net for segmentation, and a user-friendly GUI for end-to-end interaction.

Overview
Classification Model: Modified ResNet-18 tailored for brain tumor image classification.

Segmentation Model: U-Net architecture trained on the BraTS2020 dataset.

Explainability: Grad-CAM visualizations to highlight tumor regions influencing predictions.

GUI: An intuitive graphical interface for uploading images, viewing predictions, and visualizing heatmaps.

Features
MRI-based brain tumor classification (e.g., glioma, meningioma, pituitary, or no tumor).

Pixel-level tumor segmentation output.

Grad-CAM overlays for model interpretability.

Interactive interface built with Flask.

Clean modular code structure for easy extension.


Datasets
Classification: Brain Tumor dataset from Kaggle

Segmentation: BraTS2020 dataset (T1ce modality used)



Results
Classification Accuracy: 98.12%

Segmentation Accuracy: 98% IoU 

Interpretability: Qualitative heatmap overlays via Grad-CAM

Future Work
Incorporating Reinforcement Learning from Human Feedback (RLHF) for model refinement using expert evaluations.

Improving segmentation performance with hybrid attention mechanisms.

Expanding GUI functionalities for batch analysis and report generation.
