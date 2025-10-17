import sys
from functions import TiffImageInfo 

IMAGE_PATH = sys.argv[1]

GAMMA_VALUES_TO_TEST = [0.3, 0.5, 1.0, 1.2, ] 

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python intensity_transform_gamma.py <filne_name.tif> ")
        sys.exit(1)

    img = TiffImageInfo(IMAGE_PATH)

    if img.tif:
        img.visualize_gamma_effect(IMAGE_PATH, GAMMA_VALUES_TO_TEST)
    else:
        print(f"Cannot run visualization. TIFF file handle for {IMAGE_PATH} is closed or invalid.")
