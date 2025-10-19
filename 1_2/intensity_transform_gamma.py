import sys
from functions import TiffImageInfo


GAMMA_VALUES_TO_TEST = [0.3, 0.5, 1.5, 3]


def main():
    if len(sys.argv) != 2:
        print("Usage: python intensity_transform_gamma.py <file_name.tif>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    img = TiffImageInfo(IMAGE_PATH)

    if img.tif:
        img.visualize_gamma_effect(IMAGE_PATH, GAMMA_VALUES_TO_TEST)
    else:
        print(f"Cannot run visualization. TIFF file handle for {IMAGE_PATH} is closed or invalid.")


if __name__ == "__main__":
    main()
