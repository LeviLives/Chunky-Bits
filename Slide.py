import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import time

# Configure matplotlib for better image display
rcParams['figure.figsize'] = (10, 10)
plt.style.use('seaborn-v0_8-darkgrid')


def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """Load image, resize with aspect ratio, add padding, return RGB and grayscale versions"""
    img_rgb = Image.open(image_path).convert('RGB')
    img_gray = img_rgb.convert('L')  # Grayscale for brightness analysis

    original_width, original_height = img_rgb.size
    target_width, target_height = target_size

    # Calculate scaling to maintain aspect ratio
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_factor = min(width_ratio, height_ratio)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize with high-quality resampling
    resized_rgb = img_rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_gray = img_gray.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create padded images with black background
    padded_rgb = Image.new('RGB', target_size, color=(0, 0, 0))
    padded_gray = Image.new('L', target_size, color=0)

    # Center the resized image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    padded_rgb.paste(resized_rgb, (paste_x, paste_y))
    padded_gray.paste(resized_gray, (paste_x, paste_y))

    return np.array(padded_rgb), np.array(padded_gray), (paste_x, paste_y, new_width, new_height)


def split_into_chunks(image_array, chunk_size):
    """Split image array into chunks and return chunks with their positions"""
    chunks = []
    positions = []

    if len(image_array.shape) == 3:
        height, width, _ = image_array.shape
    else:
        height, width = image_array.shape

    num_chunks_x = width // chunk_size
    num_chunks_y = height // chunk_size

    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            x_start = x * chunk_size
            x_end = x_start + chunk_size
            y_start = y * chunk_size
            y_end = y_start + chunk_size

            chunk = image_array[y_start:y_end, x_start:x_end]
            chunks.append(chunk)
            positions.append((x_start, y_start, x_end, y_end))

    return chunks, positions, (num_chunks_x, num_chunks_y)


def calculate_brightness(chunk):
    """Calculate average brightness of a grayscale chunk"""
    return np.mean(chunk)


def transform_image(input_path, reference_path, output_path, chunk_size=8):
    """Transform input image to match reference brightness pattern while preserving color"""
    # Load and preprocess images
    input_rgb, input_gray, _ = load_and_preprocess_image(input_path)
    _, reference_gray, _ = load_and_preprocess_image(reference_path)

    # Split into chunks
    input_rgb_chunks, _, _ = split_into_chunks(input_rgb, chunk_size)
    input_gray_chunks, _, _ = split_into_chunks(input_gray, chunk_size)
    reference_gray_chunks, reference_positions, _ = split_into_chunks(reference_gray, chunk_size)

    # Analyze brightness
    input_brightness = [calculate_brightness(chunk) for chunk in input_gray_chunks]
    reference_brightness = [calculate_brightness(chunk) for chunk in reference_gray_chunks]

    # Sort chunks by brightness
    sorted_input_indices = np.argsort(input_brightness)
    sorted_input_rgb_chunks = [input_rgb_chunks[i] for i in sorted_input_indices]

    sorted_reference_indices = np.argsort(reference_brightness)
    sorted_reference_positions = [reference_positions[i] for i in sorted_reference_indices]

    # Generate output
    output_array = np.zeros_like(input_rgb)
    for rgb_chunk, position in zip(sorted_input_rgb_chunks, sorted_reference_positions):
        x_start, y_start, x_end, y_end = position
        output_array[y_start:y_end, x_start:x_end] = rgb_chunk

    # Save and return
    output_image = Image.fromarray(output_array.astype(np.uint8))
    output_image.save(output_path)
    return output_image


def run_slideshow(image_paths, delay=2):
    """Display images in a slideshow with specified delay between frames (seconds)"""
    fig, ax = plt.subplots()
    plt.axis('off')  # Hide axes for cleaner display

    # Function to update the display with next image
    def update(frame):
        ax.clear()
        ax.axis('off')
        img = Image.open(image_paths[frame])
        ax.imshow(img)
        ax.set_title(f"Chunk Size: {64 // (2 ** frame)}x{64 // (2 ** frame)}", fontsize=14)
        # Might change this to display proper title chunk size, fix
        return ax

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(image_paths),
        interval=delay * 1000,  # Convert seconds to milliseconds
        repeat=False
    )

    plt.show()


if __name__ == "__main__":
    # Configuration
    INPUT_IMAGE = "color.png"
    REFERENCE_IMAGE = "Star.jpg"
    # INPUT_IMAGE = "I_like_you.png"
    # REFERENCE_IMAGE = "nerd.jpg"
    OUTPUT_PREFIX = "output_chunk_"  # Prefix for generated images
    CHUNK_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]  # Sequential chunk sizes
    SLIDESHOW_DELAY = 2  # Seconds between images

    # Validate input files
    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: Input image '{INPUT_IMAGE}' not found.")
        exit(1)
    if not os.path.exists(REFERENCE_IMAGE):
        print(f"Error: Reference image '{REFERENCE_IMAGE}' not found.")
        exit(1)

    # Generate images for each chunk size
    output_paths = []
    print("Generating images with different chunk sizes...")
    for size in CHUNK_SIZES:
        output_path = f"{OUTPUT_PREFIX}{size}.jpg"
        print(f"Processing chunk size {size}x{size}...")
        transform_image(INPUT_IMAGE, REFERENCE_IMAGE, output_path, chunk_size=size)
        output_paths.append(output_path)

    # Display slideshow
    print("Starting slideshow...")
    run_slideshow(output_paths, delay=SLIDESHOW_DELAY)

    print("Process complete! All images saved with prefix:", OUTPUT_PREFIX)
