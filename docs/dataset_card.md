# Dataset Card: Saferoute Traffic Sign Dataset

## Dataset Description
The Saferoute dataset is a custom collection designed for traffic sign recognition. It is a combined dataset utilizing images from **PauloLab** and the **German Traffic Sign Recognition Benchmark (GTSRB)**.

## Dataset Structure
* **Total Classes:** 10 (Filtered down from the original 16 classes to focus on the most critical signs for this sprint).
* **Sources:** 
  * PauloLab
  * GTSRB (Supplemental variations)
* **Task:** Object Detection and Image Classification (utilizing CNN and YOLO architectures in a cascade design).

## Intended Use
This dataset is strictly used for Project Sprint 1 of the Saferoute application to train baseline models for traffic sign detection.
