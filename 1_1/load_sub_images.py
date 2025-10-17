from tif_functions import TiffImageInfo
import sys

def process_sub_images(filename):
    tif_info = TiffImageInfo(filename) 
    
    if not tif_info.tif:
        print(f"Failed to open TIFF file: {filename}")
        return

    dir_count = 0

    while tif_info.tif.SetDirectory(dir_count):
        print(tif_info.tif.SetDirectory)
        print(f"\nDirectory {dir_count} ")
        tif_info.print_image_info()
        # tif_info.visualise_image()
        dir_count += 1

    if dir_count == 0:
        print("Other TIFF directories was not found.")

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python code.py <tiff_filename>")
        sys.exit(1)
    
    process_sub_images(sys.argv[1])
