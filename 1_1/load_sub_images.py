from tif_functions import TiffImageInfo
import sys

def main():

    if len(sys.argv) < 2:
        print("Usage: python load_image.py <tiff_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    tif_info = TiffImageInfo(filename)

    if tif_info.tif:
        tif_info.process_sub_images(filename)

if __name__ == "__main__":
    main()
