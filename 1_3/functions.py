from libtiff import TIFF
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

class TiffImageInfo:

    def __init__(self, filename:str):
        self.filename = filename
        try:
            self.tif = TIFF.open(filename)
        except Exception as e:
            print(f"Error occured opening tiff file: {filename}")
            self.tif = None
    
    def __del__(self):
        if self.tif:
            self.tif.close()

    def to_float_image(self, img_uint8: np.ndarray) -> np.ndarray:
        """
        Convert 8-bit unsigned integer image to float [0, 1].
        Used for intermediate processing.
        """
        return img_uint8.astype(np.float32) / 255.0

    def to_uint8_image(self, img_float: np.ndarray) -> np.ndarray:
        """
        Convert float image [0, 1] (or any float range) back to 8-bit [0, 255].
        Used for displaying or saving.
        """
        img_scaled = np.clip(img_float * 255.0, 0, 255)
        return img_scaled.astype(np.uint8)

    def apply_spatial_filter(self, img_float: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply a spatial filter (convolution) on a float image.
        """
        # Get image dimensions
        height, width = img_float.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        # Pad image to handle borders
        padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        # Prepare output
        filtered = np.zeros_like(img_float)

        # Convolution
        for i in range(height):
            for j in range(width):
                region = padded_img[i:i+k_h, j:j+k_w]
                filtered[i, j] = np.sum(region * kernel)

        # Clip to valid range
        return np.clip(filtered, 0.0, 1.0)

    def visualize_blur(self, img_array: np.ndarray):
        """
        Demonstrate spatial blurring using a mean filter.
        """
        img_float = self.to_float_image(img_array)

        # Define a simple 3x3 averaging kernel
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0

        # Apply filter
        blurred = self.apply_spatial_filter(img_float, kernel)
        blurred_uint8 = self.to_uint8_image(blurred)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(blurred_uint8, cmap='gray')
        axes[1].set_title("Blurred Image (3×3 Mean Filter)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
