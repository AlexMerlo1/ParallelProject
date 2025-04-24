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
import warnings
# ignore the error :)
warnings.filterwarnings("ignore", message=".*out-of-thread context could not be cleaned up.*")


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

    # help clean up the gpu: prevent out-of-thread error
    cuda.Context.synchronize()
    d_image.free()

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
         # contains 101,000 images: 101 categories with 1000 images each 
        dataset_path = kagglehub.dataset_download("kmader/food41")
    else:
        print("Dataset already exists. Using the existing dataset.")

    # warm_up_gpu()

    # apple_pie has 1000 images
    image_folder = os.path.join(dataset_path, "images", "apple_pie")
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg"))]

    # Max threads per block: 1024
    # Max block dim x: 1024
    # Max block dim y: 1024
    # Max block dim z: 64

    block_sizes = [
        
        # small numbers
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),

        # maximums in each dimension
        (1, 1, 64),
        (1, 1024, 1),
        (1024, 1, 1),

        (4, 4, 1),
        (8, 8, 1),
        (16, 16, 1),

        # most cube like
        (16, 8, 8), # 16 x 8 x 8 = 1024

        (32, 32, 1), # 32 x 32 = 1024
        (16, 64, 1), # 16 x 64 = 1024
        (8, 128, 1), # 8 x 128 = 1024
        (4, 256, 1), # 4 x 256 = 1024
        (2, 512, 1), # 2 x 512 = 1024
        
        (1024, 1, 1),
        (512, 1, 1),
        (256, 1, 1),
        (128, 1, 1),
        (64, 1, 1),
        (32, 1, 1),
        (16, 1, 1),
        (8, 1, 1)
    ]

    batch_sizes = [
        # 10,
        # 25,
        50,
        1, # 1 image per batch
        1000 # all images in 1 bath
    ]

    global_times = {block_size: [] for block_size in block_sizes}
    avg_time_per_batch_size = {}
    batch_size_times = {batch_size: [] for batch_size in batch_sizes}

    # for each batch size
    for batch_size in batch_sizes:
        print(f"\nRunning tests for batch size = {batch_size}")
        times = {block_size: [] for block_size in block_sizes}
        times_avg = {}

        # for each block size
        for block_size in block_sizes:

            # run multiple times per block size
            for run in range(5):
                start_time = time.time()
                
                # for all 1000 images
                for i in range(0, len(image_paths), batch_size):
                    batch_images = image_paths[i:i + batch_size]
                    process_batch(batch_images, block_size=block_size)
                
                end_time = time.time()
                times[block_size].append(end_time - start_time)
                global_times[block_size].append(end_time - start_time)
                batch_size_times[batch_size].append(end_time - start_time)

            times_avg[block_size] = sum(times[block_size]) / len(times[block_size])
            avg_time_for_this_batch = sum(times_avg.values()) / len(times_avg)
            avg_time_per_batch_size[batch_size] = avg_time_for_this_batch

            print(f"Block {block_size}: Avg Time = {times_avg[block_size]:.2f}s")

        # Find and print min and max average time for this batch size
        min_block = min(times_avg, key=times_avg.get)
        max_block = max(times_avg, key=times_avg.get)
        print(f"\n[Batch Size = {batch_size}] Fastest Block Size: {min_block} with Avg Time = {times_avg[min_block]:.2f}s")
        print(f"[Batch Size = {batch_size}] Slowest Block Size: {max_block} with Avg Time = {times_avg[max_block]:.2f}s")

        # we don't really need to show a chart for each batch size
        '''
        # Boxplot for each batch size
        block_labels = [f"{b[0]}x{b[1]}x{b[2]}" for b in block_sizes]
        time_data = [times[b] for b in block_sizes]

        plt.figure(figsize=(12, 6))
        plt.boxplot(time_data, tick_labels=block_labels)
        plt.xlabel('Block Size (threads per block)')
        plt.ylabel('Total Processing Time per 1000 images (seconds)')
        plt.title(f'Box Plot: Processing Time vs Block Size (Batch Size = {batch_size})')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Barchart average time for each batch size
        avg_times = [times_avg[b] for b in block_sizes]
        plt.figure(figsize=(12, 5))
        plt.bar(block_labels, avg_times)
        plt.xlabel('Block Size (threads per block)')
        plt.ylabel('Total Processing Time per 1000 images (seconds)')
        plt.title(f'Average Processing Time vs Block Size (Batch Size = {batch_size})')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
        '''

    # After everything is processed

    # block sizes

    # block size vs time (all batch sizes)
    block_labels = [f"{b[0]}x{b[1]}x{b[2]}" for b in block_sizes]
    time_data = [global_times[b] for b in block_sizes]

    plt.figure(figsize=(12, 6))
    plt.boxplot(time_data, tick_labels=block_labels)
    plt.xlabel('Block Size (threads per block)')
    plt.ylabel('Total Processing Time per 1000 images (seconds)')
    plt.title('Processing Time vs Block Size (All Batch Sizes)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Fastest and Slowest Block Sizes Across All Batches ---
    block_avg_times = {block: sum(times) / len(times) for block, times in global_times.items()}

    min_block = min(block_avg_times, key=block_avg_times.get)
    max_block = max(block_avg_times, key=block_avg_times.get)

    print("\n=== Block Size Summary Across All Batches ===")
    print(f"Fastest Block Size: {min_block} with Avg Time = {block_avg_times[min_block]:.2f}s")
    print(f"Slowest Block Size: {max_block} with Avg Time = {block_avg_times[max_block]:.2f}s")


    # batch sizes

    # batch size vs time
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [batch_size_times[b] for b in batch_sizes],
        tick_labels=[str(b) for b in batch_sizes]
    )
    plt.xlabel('Batch Size')
    plt.ylabel('Total Processing Time per 1000 images (seconds)')
    plt.title('Processing Time Distribution Across Batch Sizes')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Fastest and Slowest Batch Sizes ---
    min_batch = min(avg_time_per_batch_size, key=avg_time_per_batch_size.get)
    max_batch = max(avg_time_per_batch_size, key=avg_time_per_batch_size.get)

    print("\n=== Batch Size Summary ===")
    print(f"Fastest Batch Size: {min_batch} with Avg Time = {avg_time_per_batch_size[min_batch]:.2f}s")
    print(f"Slowest Batch Size: {max_batch} with Avg Time = {avg_time_per_batch_size[max_batch]:.2f}s")


if __name__ == "__main__":
    main()


'''
data:
Running tests for batch size = 50
Block (1, 1, 1): Avg Time = 3.64s
Block (2, 1, 1): Avg Time = 3.21s
Block (1, 2, 1): Avg Time = 3.23s
Block (1, 1, 2): Avg Time = 3.63s
Block (1, 1, 64): Avg Time = 3.59s
Block (1, 1024, 1): Avg Time = 3.07s
Block (1024, 1, 1): Avg Time = 2.86s
Block (4, 4, 1): Avg Time = 3.02s
Block (8, 8, 1): Avg Time = 2.95s
Block (16, 16, 1): Avg Time = 2.83s
Block (16, 8, 8): Avg Time = 2.99s
Block (32, 32, 1): Avg Time = 2.92s
Block (16, 64, 1): Avg Time = 2.97s
Block (8, 128, 1): Avg Time = 2.89s
Block (4, 256, 1): Avg Time = 2.87s
Block (2, 512, 1): Avg Time = 3.02s
Block (1024, 1, 1): Avg Time = 2.91s
Block (512, 1, 1): Avg Time = 2.88s
Block (256, 1, 1): Avg Time = 2.83s
Block (128, 1, 1): Avg Time = 2.89s
Block (64, 1, 1): Avg Time = 2.88s
Block (32, 1, 1): Avg Time = 2.95s
Block (16, 1, 1): Avg Time = 2.88s
Block (8, 1, 1): Avg Time = 3.01s

[Batch Size = 50] Fastest Block Size: (256, 1, 1) with Avg Time = 2.83s
[Batch Size = 50] Slowest Block Size: (1, 1, 1) with Avg Time = 3.64s

Running tests for batch size = 1
Block (1, 1, 1): Avg Time = 3.63s
Block (2, 1, 1): Avg Time = 3.21s
Block (1, 2, 1): Avg Time = 3.17s
Block (1, 1, 2): Avg Time = 3.70s
Block (1, 1, 64): Avg Time = 3.48s
Block (1, 1024, 1): Avg Time = 3.01s
Block (1024, 1, 1): Avg Time = 2.98s
Block (4, 4, 1): Avg Time = 2.95s
Block (8, 8, 1): Avg Time = 2.88s
Block (16, 16, 1): Avg Time = 2.88s
Block (16, 8, 8): Avg Time = 3.16s
Block (32, 32, 1): Avg Time = 2.91s
Block (16, 64, 1): Avg Time = 2.90s
Block (8, 128, 1): Avg Time = 2.88s
Block (4, 256, 1): Avg Time = 3.06s
Block (2, 512, 1): Avg Time = 3.07s
Block (1024, 1, 1): Avg Time = 2.93s
Block (512, 1, 1): Avg Time = 2.82s
Block (256, 1, 1): Avg Time = 2.87s
Block (128, 1, 1): Avg Time = 3.01s
Block (64, 1, 1): Avg Time = 2.90s
Block (32, 1, 1): Avg Time = 2.86s
Block (16, 1, 1): Avg Time = 2.90s
Block (8, 1, 1): Avg Time = 3.03s

[Batch Size = 1] Fastest Block Size: (512, 1, 1) with Avg Time = 2.82s
[Batch Size = 1] Slowest Block Size: (1, 1, 2) with Avg Time = 3.70s

Running tests for batch size = 1000
Block (1, 1, 1): Avg Time = 3.62s
Block (2, 1, 1): Avg Time = 3.17s
Block (1, 2, 1): Avg Time = 3.29s
Block (1, 1, 2): Avg Time = 3.59s
Block (1, 1, 64): Avg Time = 3.56s
Block (1, 1024, 1): Avg Time = 3.03s
Block (1024, 1, 1): Avg Time = 2.93s
Block (4, 4, 1): Avg Time = 2.92s
Block (8, 8, 1): Avg Time = 2.94s
Block (16, 16, 1): Avg Time = 2.82s
Block (16, 8, 8): Avg Time = 3.01s
Block (32, 32, 1): Avg Time = 2.90s
Block (16, 64, 1): Avg Time = 2.87s
Block (8, 128, 1): Avg Time = 2.94s
Block (4, 256, 1): Avg Time = 2.91s
Block (2, 512, 1): Avg Time = 3.00s
Block (1024, 1, 1): Avg Time = 2.90s
Block (512, 1, 1): Avg Time = 2.89s
Block (256, 1, 1): Avg Time = 2.92s
Block (128, 1, 1): Avg Time = 2.94s
Block (64, 1, 1): Avg Time = 2.86s
Block (32, 1, 1): Avg Time = 2.82s
Block (16, 1, 1): Avg Time = 2.95s
Block (8, 1, 1): Avg Time = 3.03s

[Batch Size = 1000] Fastest Block Size: (32, 1, 1) with Avg Time = 2.82s
[Batch Size = 1000] Slowest Block Size: (1, 1, 1) with Avg Time = 3.62s


=== Block Size Summary Across All Batches ===
Fastest Block Size: (16, 16, 1) with Avg Time = 2.84s
Slowest Block Size: (1, 1, 2) with Avg Time = 3.64s


=== Batch Size Summary ===
Fastest Batch Size: 1000 with Avg Time = 3.04s
Slowest Batch Size: 1 with Avg Time = 3.05s
'''