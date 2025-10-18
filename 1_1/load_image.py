import sys
from tif_functions import TiffImageInfo

def main():

    if len(sys.argv) < 2:
        print("Usage: python load_image.py <tiff_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    tif_info = TiffImageInfo(filename)

    if tif_info.tif:
        tif_info.print_image_info()
        tif_info.visualise_image()

    sys.exit(0)

if __name__ == "__main__":
    main()