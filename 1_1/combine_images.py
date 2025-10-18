import sys
from tif_functions import TiffImageInfo

output = sys.argv[1]
image_a = 'data/imgset1/Region_001_FOV_00041_Acridine_Or_Gray.tif'
image_b = 'data/imgset1/Region_001_FOV_00041_DAPI_Gray.tif'
image_c = 'data/imgset1/Region_001_FOV_00041_FITC_Gray.tif'


def main():
    if len(sys.argv) != 2:
        print("Usage: python combine_images.py <output_file_name.tif> ")
        sys.exit(1)

    output_filename = sys.argv[1]
    
    TiffImageInfo.combine_images(image_a, image_b, image_c, output_filename)


if __name__ == "__main__":
    main()    
    