import sys
from functions import TiffImageInfo

if len(sys.argv) != 2:
    print("Usage: python blur_image.py <filename.tif>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
img = TiffImageInfo(IMAGE_PATH)
img_array = img.read_image_data()

if img_array is not None:
    img.visualize_blur(img_array)
else:
    print("Error reading image.")
