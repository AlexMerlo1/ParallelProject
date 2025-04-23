# !pip install pycuda
# !apt-get install -y nvidia-cuda-toolkit

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes the CUDA context in Colab
from pycuda.compiler import SourceModule
import kagglehub
import os
import time
import matplotlib.pyplot as plt
def warm_up_gpu(iterations=10, image_size=(256, 256)):
    kernel_code = """
    __global__ void process_image_kernel(unsigned char *d_image, int width, int height, int channels) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = (y * width + x) * channels;

            // Convert to grayscale by averaging the RGB values
            unsigned char gray_value = (d_image[idx] + d_image[idx + 1] + d_image[idx + 2]) / 3;
            d_image[idx] = gray_value;     // Red channel (after grayscale)
            d_image[idx + 1] = gray_value; // Green channel (after grayscale)
            d_image[idx + 2] = gray_value; // Blue channel (after grayscale)
        }
    }
    """

    print("Warming up GPU...")

    height, width = image_size
    channels = 3
    image_data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8).flatten()

    # Allocate device memory once
    d_image = cuda.mem_alloc(image_data.nbytes)

    # Compile kernel
    mod = SourceModule(kernel_code)
    process_image_kernel = mod.get_function("process_image_kernel")

    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])

    for i in range(iterations):
        # Optionally regenerate the image data each time for variability
        image_data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8).flatten()
        cuda.memcpy_htod(d_image, image_data)

        process_image_kernel(d_image, np.int32(width), np.int32(height), np.int32(channels),
                             block=block_size, grid=grid_size)

    cuda.Context.synchronize()
    print(f"GPU warm-up complete after {iterations} iterations.")

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes the CUDA context in Colab
from pycuda.compiler import SourceModule
import kagglehub
import os
import time
import matplotlib.pyplot as plt




def convert_image_to_matrix_cuda(image_path, block_size=(16, 16, 1)):
    # CUDA kernel to process the image (convert to grayscale)
    kernel_code = """
    __global__ void process_image_kernel(unsigned char *d_image, int width, int height, int channels) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = (y * width + x) * channels;

            // Convert to grayscale by averaging the RGB values
            unsigned char gray_value = (d_image[idx] + d_image[idx + 1] + d_image[idx + 2]) / 3;
            d_image[idx] = gray_value;     // Red channel (after grayscale)
            d_image[idx + 1] = gray_value; // Green channel (after grayscale)
            d_image[idx + 2] = gray_value; // Blue channel (after grayscale)
        }
    }
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)  # Read the image in color (BGR)
    if image is None:
        print(f"Error loading image at {image_path}!")
        return None

    # Get image dimensions
    height, width, channels = image.shape
    image_size = width * height * channels

    # Flatten the image into a 1D array
    image_data = image.flatten().astype(np.uint8)

    # Allocate memory on the device (GPU)
    d_image = cuda.mem_alloc(image_data.nbytes)

    # Copy the image data from host to device
    cuda.memcpy_htod(d_image, image_data)

    # Compile the kernel
    mod = SourceModule(kernel_code)
    process_image_kernel = mod.get_function("process_image_kernel")

    # Define block and grid dimensions for kernel launch
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])

    # Launch the kernel to process the image (convert to grayscale)
    process_image_kernel(d_image, np.int32(width), np.int32(height), np.int32(channels),
                         block=block_size, grid=grid_size)

    # Copy the result back to host memory
    cuda.memcpy_dtoh(image_data, d_image)

    # Reshape the processed data into the original image dimensions
    processed_image = image_data.reshape((height, width, channels))

    # Save the processed image (grayscale)
    output_path = image_path.replace(".jpg", "_processed.jpg")
    cv2.imwrite(output_path, processed_image)
    return output_path

def process_batch(batch_images, block_size):

    processed_images = []
    # Process images in the batch linearly (or non-parallel way) using a for loop
    for image_path in batch_images:
        processed_image = convert_image_to_matrix_cuda(image_path, block_size)
        if processed_image:
            processed_images.append(processed_image)





def main():
    dataset_path = '/content/food41'

    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading...")
        dataset_path = kagglehub.dataset_download("kmader/food41")
    else:
        print("Dataset already exists. Using the existing dataset.")

    warm_up_gpu()

    image_folder = os.path.join(dataset_path, "images", "apple_pie")
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg"))]

    block_sizes = [
        (1, 1, 1), (2, 2, 1), (4, 4, 1), (8, 8, 1), (16, 16, 1), (32, 32, 1),
        (16, 64, 1), (8, 128, 1), (4, 256, 1), (2, 512, 1), (1, 1024, 1),
        (512, 1, 1), (256, 1, 1), (128, 1, 1), (64, 1, 1), (32, 1, 1), (16, 1, 1), (8, 1, 1),
    ]

    batch_sizes = [10,25,50]

    for batch_size in batch_sizes:
        print(f"\nRunning tests for batch size = {batch_size}")
        times = {block_size: [] for block_size in block_sizes}
        times_avg = {}

        for block_size in block_sizes:
            for run in range(5):
                start_time = time.time()
                for i in range(0, len(image_paths), batch_size):
                    batch_images = image_paths[i:i + batch_size]
                    process_batch(batch_images, block_size=block_size)
                end_time = time.time()
                times[block_size].append(end_time - start_time)
            times_avg[block_size] = sum(times[block_size]) / len(times[block_size])
            print(f"Block {block_size}: Avg Time = {times_avg[block_size]:.2f}s")

        # Plot boxplot
        block_labels = [f"{b[0]}x{b[1]}" for b in block_sizes]
        time_data = [times[b] for b in block_sizes]

        plt.figure(figsize=(12, 6))
        plt.boxplot(time_data, labels=block_labels)
        plt.xlabel('Block Size (threads per block)')
        plt.ylabel('Processing Time (seconds)')
        plt.title(f'Box Plot: Processing Time vs Block Size (Batch Size = {batch_size})')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot average time bar chart
        avg_times = [times_avg[b] for b in block_sizes]
        plt.figure(figsize=(12, 5))
        plt.bar(block_labels, avg_times)
        plt.xlabel('Block Size (threads per block)')
        plt.ylabel('Average Processing Time (seconds)')
        plt.title(f'Average Processing Time vs Block Size (Batch Size = {batch_size})')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()
