import sys
from functions import TiffImageInfo
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python conversion_test.py <filename.tif>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
img = TiffImageInfo(IMAGE_PATH)
img_array = img.read_image_data()

if img_array is not None:
    # Convert to float and back
    img_float = img.to_float_image(img_array)
    img_back_to_uint8 = img.to_uint8_image(img_float)

    # Print stats
    print("Original dtype:", img_array.dtype)
    print("Float image dtype:", img_float.dtype, "| min:", img_float.min(), "max:", img_float.max())
    print("Reconverted dtype:", img_back_to_uint8.dtype)

    # Show all three for comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title("Original (uint8)")
    axes[0].axis('off')

    axes[1].imshow(img_float, cmap='gray')
    axes[1].set_title("Float [0, 1]")
    axes[1].axis('off')

    axes[2].imshow(img_back_to_uint8, cmap='gray')
    axes[2].set_title("Converted back to uint8")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not read image data.")
