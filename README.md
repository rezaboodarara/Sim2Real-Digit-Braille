# Sim2Real-Digit-Braille

This repository contains the code, data generation pipelines, and model architectures for the research project "Tactile Braille Recognition Using Sim2Real Transfer". 

## Overview

The ability to programmatically read Braille has significant implications for assistive technology and robotic perception. However, training data-driven classifiers for vision-based tactile sensors is heavily bottlenecked by the high cost and impracticality of manual real-world data collection. 

To overcome this, this project introduces a depth-based simulation-to-reality (sim-to-real) pipeline utilizing the high-resolution DIGIT tactile sensor. By training exclusively on synthetic depth data with geometrically constrained augmentations, our final Convolutional Neural Network (CNN) achieves a zero-shot transfer accuracy of 99.11% across all 26 Braille classes on unseen real-world data. 

---

## The Sim2Real Pipeline

The project is structured around a four-stage pipeline:

* **1. Object Generation:** Blender scripts automatically generate a number of 3D objects with different features. Specifically, the pipeline generated over 105,000 diverse 3D meshes of the 26 Braille letters to serve as implicit domain randomization. The algorithm maps each letter to a 3x3 spatial grid and varies physical surface features such as point geometry, sizes, and spacing.
* **2. Tacto Simulation (Depth Generation):** The system generates synthetic depth images of the 3D models using the TACTO simulator. This open-source simulator renders realistic contact images from 3D object meshes at a negligible cost.
* **3. Model Training:** The CNN trains on a batched mixture of synthetic TACTO depth images and augmented data. 
* **4. Evaluation on Real-World Data:** The final model predicts top outcomes using strictly isolated, unseen real-world depth images captured by the physical DIGIT sensor.

---

## Key Methodology and Findings

### Modality Shift: RGB vs. Depth
* Models trained purely on simulated RGB tactile images struggle to generalize to real-world data. 
* A baseline Decision Tree classifier trained exclusively on TACTO-simulated RGB images achieved only 32.0% accuracy on real-world DIGIT images.
* Switching to depth-based representations largely eliminates the domain gap. Depth maps capture the underlying surface geometry, which is more faithfully reproduced by TACTO's physics-based simulation.

### Constrained Data Augmentation
* Standard image augmentation practices, such as unconstrained rotation, are inappropriate for Braille. 
* Aggressive rotational augmentation risks generating training samples that are indistinguishable from other letter classes, introducing label noise.
* The final training regime implemented geometrically constrained, Braille-aware data augmentation to prevent the model from confusing directionally dependent letters (like 'b' and 'd') under aggressive rotation.

---

## Project Structure & Key Scripts

The repository includes several custom scripts designed to automate the sim-to-real workflow:

* **`digitInterface_Automated.py`**: An automated data collection script used to capture, label, and sequentially number real-world image files using the physical DIGIT sensor.
* **`addObjectScript.py`**: A Blender Python script used to systematically batch-generate all 26 Braille letter meshes.
* **`mainProcess.py`**: The primary script that systematically loads each letter's `.obj` file into the TACTO simulation environment.
* **`Template.py`**: An adapted simulation script that generates depth maps and computes 3D surface normals from the depth array.
* **`new2_depth_org.py`**: A custom script that eliminates `rospy` dependencies and extracts depth images directly from the real sensor.
* **`cnn_with_limited_augmentation.py`**: The training script for the final CNN model, implementing strict train-test isolation and geometrically constrained augmentation parameters.
* **`best_model_limited.keras`**: The final trained model weights that achieved peak performance.

---

## Results and Performance

The final CNN model outperforms previous diffusion-model-based baselines under a strict zero-shot evaluation protocol (sim-trained, real-tested, no fine-tuning).

| Model Iteration | Pipeline & Methodology | Accuracy |
| :--- | :--- | :--- |
| **Sim-to-Real Baseline** | TACTO Sim (RGB Modality, 3 Classes) | 32.0% |
| **Basic CNN** | 3 Conv blocks, Flatten, No data augmentation | 58.2% |
| **Tactile Diffusion Baseline** | Zero-shot baseline using a tactile diffusion model | 75.74% |
| **Final CNN Model** | Depth modality, CNN with geometrically constrained augmentation | **99.11%** |
