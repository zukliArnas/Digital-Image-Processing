from tif_functions import TiffImageInfo
import sys

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python code.py <tiff_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    tif_info = TiffImageInfo(filename) 

    if tif_info.tif:
        tif_info.print_image_info()
        tif_info.visualise_image()
