from tif_functions import TiffImageInfo
import sys
from matplotlib import pyplot as plt

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Usage: python load_sub_region.py <tiff_filename> <x1> <y1> <x2> <y2>")
        print("Example: python load_sub_region.py Kidney2.svs 0 0 511 511")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        x1 = int(sys.argv[2])
        y1 = int(sys.argv[3])
        x2 = int(sys.argv[4])
        y2 = int(sys.argv[5])
    except ValueError:
        print("Error: Coordinates (x1, y1, x2, y2) must be integers.")
        sys.exit(1)

    tif_info = TiffImageInfo(filename)


    if tif_info.tif:
        sub_image = tif_info.read_sub_region(x1, y1, x2, y2)
        if sub_image is not None:
            if tif_info.is_gray_scale(sub_image):
                plt.imshow(sub_image, cmap='gray', interpolation='nearest', 
                           vmin=0, vmax=255)
            else:
                plt.imshow(sub_image, interpolation='nearest', 
                           vmin=0, vmax=255)

            plt.title(f"Subregion [{x1}, {y1}] to [{x2}, {y2}]")
            plt.axis('off')
            plt.show()
