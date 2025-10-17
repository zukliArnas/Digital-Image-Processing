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

    # # def read_sub_region(self, x1, y1, x2, y2):
    #     if not self.tif:
    #         print("Cannot read TIFF file.")
    #         return None

    #     width = self._get_field('ImageWidth')
    #     height = self._get_field('ImageLength')
    #     samples_per_pixel = self._get_field('SamplesPerPixel')

    #     # Collecting valid coordinatesq
    #     x1 = max(0 , x1)
    #     y1 = max(0 , y1)
    #     x2 = min(width - 1, x2)
    #     y2 = min(height - 1, y2)

    #     # Caclulating sub-region dimensions
    #     sub_width = x2 - x1 + 1
    #     sub_height = y2 - y1 + 1

    #     # Checking single pixel type 
    #     bits_per_sample = self._get_field('BitsPerSample')
    #     if bits_per_sample == 8:
    #         image_dtype = np.uint8
    #     elif bits_per_sample == 16:
    #         image_dtype = np.uint16
    #     else:
    #         # Default to uint8 if info is missing or unexpected
    #         image_dtype = np.uint8

    #     if samples_per_pixel > 1:
    #         sub_region = np.zeros((sub_height, sub_width, samples_per_pixel), dtype=image_dtype)
    #     else:
    #         sub_region = np.zeros((sub_height, sub_width), dtype=image_dtype)

    #     print(f"Loading sub-region ({x1} , {y1}), to ({x2} , {y2}). Shape - {sub_region.shape}")

    #     #Handling Tiled vs Stripped reading
    #     if self.is_tiled():
    #         tile_width = self._get_field('TileWidth')
    #         tile_length = self._get_field('TileLength')
    #         tiles_x_count = math.ceil(width / tile_width)

    #         x_start = x1 // tile_width
    #         y_start = y1 // tile_length
    #         x_end = x2 // tile_width
    #         y_end = y2 // tile_length

    #         for ty in range(y_start, y_end + 1):
    #             for tx in range(x_start, x_end + 1):
                    
    #                 tile_idx = ty * tiles_x_count + tx

    #                 tile_x_coord = tx * tile_width
    #                 tile_y_coord = ty * tile_length

    #                 try:
    #                     tile_data = self.tif.read_tiles(tile_idx)
    #                 except AttributeError:
    #                      tile_data = self.tif.read_tiles(tile_x_coord, tile_y_coord)
    #                      continue
                    
    #                 global_x_start = tile_x_coord # tx * tile_width
    #                 global_y_start = tile_y_coord # ty * tile_length

    #                 # Calculate overlap between tile and requested sub-region
    #                 overlap_x_start = max(x1, global_x_start)
    #                 overlap_y_start = max(y1, global_y_start)
    #                 overlap_x_end = min(x2, global_x_start + tile_width - 1)
    #                 overlap_y_end = min(y2, global_y_start + tile_length - 1)

    #                 tile_slice_x_start = overlap_x_start - global_x_start
    #                 tile_slice_y_start = overlap_y_start - global_y_start
    #                 tile_slice_x_end = tile_slice_x_start + (overlap_x_end - overlap_x_start) + 1
    #                 tile_slice_y_end = tile_slice_y_start + (overlap_y_end - overlap_y_start) + 1

    #                 # Calculate the slice in the final subregion
    #                 sub_slice_x_start = overlap_x_start - x1
    #                 sub_slice_y_start = overlap_y_start - y1
    #                 sub_slice_x_end = sub_slice_x_start + (overlap_x_end - overlap_x_start) + 1
    #                 sub_slice_y_end = sub_slice_y_start + (overlap_y_end - overlap_y_start) + 1

    #                 if samples_per_pixel > 1:
    #                     sub_region[sub_slice_y_start:sub_slice_y_end, sub_slice_x_start:sub_slice_x_end, :] = \
    #                     tile_data[tile_slice_y_start:tile_slice_y_end, tile_slice_x_start:tile_slice_x_end, :]
    #                 else:
    #                     sub_region[sub_slice_y_start:sub_slice_y_end, sub_slice_x_start:sub_slice_x_end] = \
    #                     tile_data[tile_slice_y_start:tile_slice_y_end,  tile_slice_x_start:tile_slice_x_end]
        
    #     else:
    #         # Handleing Striped images, but here we need to read whole image
    #         print("Striped images require full image load in order to crop into subregions.")
    #         full_image = self.tif.read_image()
    #         if full_image is not None:
    #             sub_region = full_image[y1:y2+1, x1:x2+1]
    #         else:
    #             return None
    #     return sub_region

    @staticmethod
    def combine_images(image_a, image_b, image_c, output):
        try:
            tif_a = TIFF.open(image_a, mode='r')
            img_a = tif_a.read_image()
            tif_a.close()

            tif_b = TIFF.open(image_b, mode='r')
            img_b = tif_b.read_image()
            tif_b.close()

            tif_c = TIFF.open(image_c, mode='r')
            img_c = tif_c.read_image()
            tif_c.close()

        except Exception as e:
            print(f"Error reading TIFF file: {e}")
            return

        if not (img_a.shape == img_b.shape == img_c.shape):
            print("Error: Images must have the same dimensions to be combined.")
            return

        combined_image = np.dstack([img_a, img_b, img_c])

        try:
            tif_out = TIFF.open(output, mode='w')
                
            # Write the image data
            tif_out.write_image(combined_image, write_rgb=True)
            tif_out.close()
        
            print(f"Successfully saved combined image")
        
        except Exception as e:
            print(f"Error saving combined image: {e}")
