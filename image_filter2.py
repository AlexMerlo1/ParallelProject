import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes the CUDA context in Colab
from pycuda.compiler import SourceModule
import kagglehub
import os
import time
import matplotlib.pyplot as plt

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

def convert_image_to_matrix_cuda(image_path, block_size=(16, 16, 1)):
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

def process_batch(batch_images, block_size=(16, 16, 1)):
    start_time = time.time()

    processed_images = []
    # Process images in the batch linearly (or non-parallel way) using a for loop
    for image_path in batch_images:
        processed_image = convert_image_to_matrix_cuda(image_path, block_size)
        if processed_image:
            processed_images.append(processed_image)

    end_time = time.time()
    # print(f"Processed batch {len(processed_images)} images in {end_time - start_time:.2f} seconds.")
    return end_time - start_time  # Return time for batch processing

def main():
    # Path to check if dataset is already downloaded
    dataset_path = '/content/food41'  # Change this to the path where your dataset is stored locally
    # now I don't have to keep redownloading everytime

    # Check if the dataset exists, if not, download it
    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading...")
        dataset_path = kagglehub.dataset_download("kmader/food41")
    else:
        print("Dataset already exists. Using the existing dataset.")

    # Generate a list of image paths to process (example for "apple_pie" folder)
    image_folder = os.path.join(dataset_path, "images", "apple_pie")
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg"))]

    block_sizes = [
        (1, 1, 1),
        (2, 2, 1),
        (4, 4, 1),
        (8, 8, 1),
        (16, 16, 1),
        (32, 32, 1),
        (16, 64, 1),
        (8, 128, 1),
        (4, 256, 1),
        (2, 512, 1),
        (1, 1024, 1),
        (8, 1, 1),
        (16, 1, 1),
        (32, 1, 1),
        (64, 1, 1),
        (128, 1, 1),
        (256, 1, 1),
        (512, 1, 1),
    ]
    times = {block_size: [] for block_size in block_sizes}

    for block_size in block_sizes:
        batch_size = 50
        for i in range(0, len(image_paths), batch_size):
            batch_images = image_paths[i:i + batch_size]
            batch_time = process_batch(batch_images, block_size=block_size)
            times[block_size].append(batch_time)
            print(f"Processed block_size: {block_size} in {batch_time:.3f}s")


    # Plotting the resulting times for differing block sizes using a box plot
    block_labels = [f"{b[0]}x{b[1]}" for b in block_sizes]
    time_data = [times[b] for b in block_sizes]

    plt.boxplot(time_data, labels=block_labels)
    plt.xlabel('Block Size (threads per block)')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time vs Block Size (Box Plot)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
