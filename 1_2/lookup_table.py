import sys
from functions import TiffImageInfo


def main():
    if len(sys.argv) != 3:
        print("Usage: python lookup_table_transform.py <filename.tif> <gamma_value>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    GAMMA = float(sys.argv[2])

    img = TiffImageInfo(IMAGE_PATH)
    img_array = img.read_image_data()

    if img_array is not None:
        img.visualize_lut_gamma(img_array, gamma=GAMMA)
    else:
        print("Error: Could not read image data.")


if __name__ == "__main__":
    main()
