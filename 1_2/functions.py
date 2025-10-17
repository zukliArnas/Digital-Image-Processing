from libtiff import TIFF
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt

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

    def read_image_data(self) -> ndarray | None:
        
        if not self.tif:
            return None
        
        try:
            arr_orig = self.tif.read_image()
            
            # --- Enforce 8-bit Grayscale ---
            
            # 1. Check for 16-bit data and stop (as requested)
            if arr_orig.dtype == np.uint16:
                print(f"Error: Image {self.filename} is 16-bit ({arr_orig.dtype}).")
                return None
            
            # 2. Handle Multi-channel (e.g., RGB) images
            if arr_orig.ndim > 2:
                print(f"Input image is multi-channel ({arr_orig.ndim} dimensions). Taking the first channel as grayscale.")
                arr_orig = arr_orig[:, :, 0]
            
            # 3. Ensure final type is 8-bit unsigned integer (np.uint8)
            if arr_orig.dtype != np.uint8:
                print(f"Warning: Converting input image from {arr_orig.dtype} to 8-bit (uint8).")
                arr_orig = arr_orig.astype(np.uint8)
                    
            return arr_orig

        except Exception as e:
            print(f"Error reading image data from {self.filename}: {e}")
            return None

    def power_law_transform(self, image_array: ndarray, gamma: float) -> ndarray:
        # Normalize
        normalized_image = image_array / 255.0
        
        # Apply Power Law Transformation
        transformed_normalized = np.power(normalized_image, gamma)

        # Denormalize: Convert float [0, 1] back to 8-bit [0, 255]
        transformed_8bit = np.clip(transformed_normalized * 255, 0, 255).astype(np.uint8)
        return transformed_8bit

    def visualize_gamma_effect(self, original_image_path, gamma_values):
        original_img_array = self.read_image_data()

        if original_img_array is None:
            print("Visualization aborted because the image is not suitable (not 8-bit grayscale).")
            return

        # Create visualization plots
        num_plots = len(gamma_values) + 1
        fig, axes = plt.subplots(num_plots, 2, figsize=(10, 4 * num_plots))

        # --- Plot Original Image and Histogram (Row 0) ---
        axes[0, 0].imshow(original_img_array, cmap='gray')
        axes[0, 0].set_title(f'Original Image ({self.filename})')
        axes[0, 0].axis('off')
        
        axes[0, 1].hist(original_img_array.flatten(), bins=256, range=[0, 256], color='gray')
        axes[0, 1].set_title('Original Histogram')
        axes[0, 1].set_xlim([0, 255])

        for i, gamma in enumerate(gamma_values, 1):
            # Apply the transformation
            arr_transformed = self.power_law_transform(original_img_array, gamma)
            
            # Image Plot
            axes[i, 0].imshow(arr_transformed, cmap='gray')
            axes[i, 0].set_title(f'Transformed Image ($\gamma={gamma}$)')
            axes[i, 0].axis('off')
            
            # Histogram Plot
            axes[i, 1].hist(arr_transformed.flatten(), bins=256, range=[0, 256], color='blue')
            axes[i, 1].set_title(f'Histogram ($\gamma={gamma}$)')
            axes[i, 1].set_xlim([0, 255])

        plt.tight_layout()
        plt.show()