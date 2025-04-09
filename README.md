# CUDA-based Image Processing with PyCUDA

### Contributors
- Alex Merlo
- Ben Braniff


## Summary

This project uses CUDA and PyCUDA to process images efficiently. The primary goal is to convert images to grayscale using a CUDA kernel for parallel computation. The code includes batch processing and performance measurement for various block sizes and visualizes the results using a box plot.

## Requirements

Before running this project, ensure the following libraries and tools are installed:

- Python 3.x
- [PyCUDA](https://documen.tician.de/pycuda/)
- [OpenCV](https://opencv.org/)
- [KaggleHub](https://github.com/kaggle/kaggle-api)
- [Matplotlib](https://matplotlib.org/)
- [NVIDIA GPU with CUDA support](https://developer.nvidia.com/cuda-toolkit)

Install the required dependencies using `pip`:

```bash
pip install pycuda opencv-python kagglehub matplotlib numpy
```

## Overview

This Python code performs the following tasks:

1. **Image Grayscale Conversion using CUDA**: The `process_image_kernel` function in CUDA processes an image to convert it into grayscale by averaging the RGB values. This process is parallelized using GPU threads.

2. **Batch Processing**: The code supports batch processing of images. You can specify the batch size, and it will process multiple images in parallel.

3. **Performance Evaluation**: It measures the time taken for image processing with different CUDA block sizes, and then plots the results to show how block size influences processing time.

4. **Dataset**: The dataset is downloaded from Kaggle if not already available locally. It uses the `kagglehub` package for downloading datasets.

## How to Use

### 1. Preparing the Dataset

- The script checks if the dataset is already downloaded. If not, it downloads the dataset using `kagglehub`.
- You can modify the `dataset_path` to specify a different location to store the dataset locally.

### 2. Image Processing

- The main processing is done in the `convert_image_to_matrix_cuda` function, which:
    - Loads an image using OpenCV.
    - Flattens the image into a 1D array for GPU processing.
    - Launches a CUDA kernel to process the image and convert it to grayscale.
    - Saves the processed image as a new file.

### 3. Batch Processing

- The `process_batch` function processes a batch of images, which are passed as a list of image paths.
- The batch processing uses different block sizes for CUDA execution and measures the time taken for each batch.
- You can modify the `batch_size` in the `main` function.

### 4. Performance Evaluation

- The code evaluates the performance of the processing using different block sizes. It tests various block sizes and measures the processing time for each batch.
- The results are displayed in a box plot using Matplotlib, comparing processing times for different block sizes.

### 5. Running the Script

Simply run the script using:

```bash
python image_filter2.py
```