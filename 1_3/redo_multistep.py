import sys
from functions import TiffImageInfo


def main():
    if len(sys.argv) != 2:
        print("Usage: python redo_multistep.py <filename.tif>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    img = TiffImageInfo(IMAGE_PATH)
    img_array = img.read_image_data()

    if img_array is not None:
        img.multistep_example_3_43(img_array)
    else:
        print("Error: Could not read image data.")


if __name__ == "__main__":
    main()
