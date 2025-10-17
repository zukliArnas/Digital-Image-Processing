import sys
from tif_functions import TiffImageInfo

output = sys.argv[1]
image_a = 'imgset1/Region_001_FOV_00041_Acridine_Or_Gray.tif'
image_b = 'imgset1/Region_001_FOV_00041_DAPI_Gray.tif'
image_c = 'imgset1/Region_001_FOV_00041_FITC_Gray.tif'

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python combine_images.py <filne_name.tif> ")
        sys.exit(1)

    TiffImageInfo.combine_images(image_a, image_b, image_c, output)
