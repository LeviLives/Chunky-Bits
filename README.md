Use slides.py with     
    INPUT_IMAGE = "color.png"
    REFERENCE_IMAGE = "Star.jpg"
Change these two to do stuff

Output will be     
    OUTPUT_PREFIX = "output_chunk_"  # Prefix for generated images
    CHUNK_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

Prefix + how ever many chunk sizes you defined. I'd suggest power of 2 for chunk sizes
