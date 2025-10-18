import sys
from functions import TiffImageInfo

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python piece_wise_transform.py <filename.tif>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    img = TiffImageInfo(IMAGE_PATH)
    img_array = img.read_image_data()

    if img_array is not None:
        img.visualize_piecewise_linear(img_array)
        img.visualize_threshold(img_array, threshold_value=120)
