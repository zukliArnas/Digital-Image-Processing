from libtiff import TIFF
import numpy as np
from matplotlib import pyplot as plt

class TiffImageInfo:
    
    def __init__(self, filename):
        self.filename = filename
        try:
            self.tif = TIFF.open(filename)
        except Exception as e:
            print(f"Error occured opening tiff file: {filename}")
            self.tif = None
    
    def __del__(self):
        if self.tif:
            self.tif.close()

    def is_gray_scale(self, image):
        return image.ndim == 2
        
    def visualise_image(self):
        
        image = self.tif.read_image()
        if image is None:
            print("No image data in TIFF file.")
            return
        
        if self.is_gray_scale(image):
            plt.imshow(image, cmap='gray', interpolation='nearest', 
                       vmin=0, vmax=255)
        else:
            plt.imshow(image, interpolation='nearest', 
                       vmin=0, vmax=255)

        plt.title(f"Image from {self.filename}")
        plt.axis('off')
        plt.show()
    
    def _get_field(self, tag_name):
        if self.tif:
            return self.tif.GetField(tag_name)
        return None

    def is_tiled(self):
        return self._get_field('TileWidth') is not None

    def print_image_info(self):
        if not self.tif:
            print("Cannot print tiff info")
            return
        print(f"\n=== TIFF Image Information for '{self.filename}' ===")

        print(f"    TIFF Directory {self.tif.CurrentDirectory()}")
        print(f"    Image Width: {self._get_field('ImageWidth')} Image Length: {self._get_field('ImageLength')} Image Depth: {self._get_field('ImageDepth')}")
        
        if self.is_tiled():
            print(f"    Storage Type: TILED")
            print(f"    Tile Width: {self._get_field('TileWidth')} Tile Length: {self._get_field('TileLength')}")
        else:
            print(f"    Storage Type: STRIPED")
            rows_per_strip = self._get_field('RowsPerStrip')
            strip_offsets = self._get_field('StripOffsets')
            print(f"    Rows Per Strip: {rows_per_strip}")
        print(f"    Depth (BitsPerSample): {self._get_field('BitsPerSample')}")
        print(f"    Samples/Pixel: {self._get_field('SamplesPerPixel')}")
        print(f"    Compression Scheme: {self._get_field('Compression')}")
        print(f"    Photometric Interpretation: {self._get_field('Photometric')}")
        print(f"    Planar Configuration: {self._get_field('PlanarConfig')}")
        print(f"    Subfile Type: {self._get_field('SubfileType')}")
        print(f"    Image Description: {self._get_field('ImageDescription')}")
        print("\n")


    @staticmethod
    def to_8bit(image_array):
        img_float = image_array.astype(np.float64)
        max_val = img_float.max()
        
        normalized_img = (img_float / max_val) * 255.0
        return normalized_img.astype(np.uint8)
    
    @staticmethod
    def combine_images(image_a, image_b, image_c, output):
        try:
            tif_r = TIFF.open(image_a, mode='r')
            img_r = TiffImageInfo.to_8bit(tif_r.read_image())
            tif_r.close()

            # DAPI (B channel)
            tif_b = TIFF.open(image_b, mode='r')
            img_b = TiffImageInfo.to_8bit(tif_b.read_image())
            tif_b.close()

            # FITC (G channel)
            tif_g = TIFF.open(image_c, mode='r')
            img_g = TiffImageInfo.to_8bit(tif_g.read_image())
            tif_g.close()
        except Exception as e:
            print(f"Error reading TIFF file: {e}")
            return

        if not (img_r.shape == img_b.shape == img_g.shape):
            print("Error: Images must have the same dimensions to be combined.")
            return

        combined_image = np.dstack([img_r, img_g, img_b])
        
        try:
            tif_out = TIFF.open(output, mode='w')
            tif_out.write_image(combined_image, write_rgb=True)
            tif_out.close()
                
        except Exception as e:
            print(f"Error saving combined image: {e}")

        print(f"Successfully saved combined image")
        return combined_image
    
    def process_sub_images(self, filename):
        
        if not self.tif:
            print(f"Failed to open TIFF file: {filename}")
            return

        dir_count = 0

        while self.tif.SetDirectory(dir_count):
            print(self.tif.SetDirectory)
            print(f"\nDirectory {dir_count} ")
            self.print_image_info()
            dir_count += 1

        if dir_count == 0:
            print("Other TIFF directories was not found.")
