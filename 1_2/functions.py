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
            axes[i, 0].set_title(rf'Transformed Image ($\gamma={gamma}$)')
            axes[i, 0].axis('off')
            
            # Histogram Plot
            axes[i, 1].hist(arr_transformed.flatten(), bins=256, range=[0, 256], color='blue')
            axes[i, 0].set_title(rf'Transformed Image ($\gamma={gamma}$)')
            axes[i, 1].set_xlim([0, 255])

        plt.tight_layout(pad=1.5, w_pad=2.0)
        plt.show()

    def transform_piece_wise(self, img: np.ndarray, r1, s1, r2, s2) -> np.ndarray:

        img = img.astype(np.float32)
        pos = [img <= r1, (img > r1) & (img <= r2), img > r2]
        result = np.piecewise(img, pos,
            [
                lambda x: (s1 / r1) * x,
                lambda x: ((s2 - s1) / (r2 - r1)) * (x - r1) + s1,
                lambda x: ((255 - s2) / (255 - r2)) * (x - r2) + s2
            ]
        )
        return np.uint8(np.clip(result, 0, 255))
    
    def visualize_piecewise_linear(self, img_array, params=None):
        """
        Visualize piecewise linear histogram stretching.
        If no params are given, automatically estimate r1 and r2
        from the 5th and 95th intensity percentiles.
        """

        # --- Optional automatic range detection ---
        if params is None:
            hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0,256])
            cdf = hist.cumsum()
            cdf_norm = cdf / cdf[-1]

            # Find where 5% and 95% of pixels fall
            r1 = np.searchsorted(cdf_norm, 0.05)
            r2 = np.searchsorted(cdf_norm, 0.95)
            s1, s2 = 0, 255
            params = (r1, s1, r2, s2)
            print(f"[Auto] Suggested stretch params: r1={r1}, s1={s1}, r2={r2}, s2={s2}")
        else:
            r1, s1, r2, s2 = params
            print(f"[Manual] Using given params: r1={r1}, s1={s1}, r2={r2}, s2={s2}")

        # --- Apply transform ---
        stretched_img = self.transform_piece_wise(img_array, r1, s1, r2, s2)
        print("Image stats  -> Min:", img_array.min(), " Max:", img_array.max())

        # --- Show results ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(stretched_img, cmap='gray')
        axes[0].set_title(f'Piecewise Linear Transform\n(r1,s1,r2,s2)=({r1},{s1},{r2},{s2})')
        axes[0].axis('off')

        axes[1].hist(stretched_img.flatten(), bins=256, range=[0, 256], color='blue')
        axes[1].set_title('Histogram (After Stretching)')
        axes[1].set_xlim([0,255])
        plt.tight_layout()
        plt.show()

    def threshold(self, img: np.ndarray, thresh: int) -> np.ndarray:
        return np.uint8((img > thresh) * 255)

    def visualize_threshold(self, img_array, threshold_value):
        thresholded_img = self.threshold(img_array, threshold_value)

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(thresholded_img, cmap='gray')
        axes[0].set_title(f'Thresholded Image (T={threshold_value})')
        axes[0].axis('off')

        axes[1].hist(thresholded_img.flatten(), bins=256, range=[0, 256], color='red')
        axes[1].set_title('Histogram (After Thresholding)')
        plt.show()

    def compute_histogram(self, img: np.ndarray) -> np.ndarray:
        hist = np.zeros(256, dtype=int)
        for value in img.flatten():
            hist[value] += 1
        return hist

    def visualize_histogram(self, img: np.ndarray):
        hist = self.compute_histogram(img)
        plt.figure(figsize=(8,4))
        plt.bar(range(256), hist, color='gray')
        plt.title("Image Histogram")
        plt.xlabel("Intensity Value")
        plt.ylabel("Pixel Count")
        plt.xlim([0,255])
        plt.show()

    def histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """
        Perform histogram equalization on a grayscale image.
        """

        # Step 1: Compute histogram
        hist, _ = np.histogram(img.flatten(), bins=256, range=[0,256])

        # Step 2: Compute CDF (Cumulative Distribution Function)
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)  # mask zeros to avoid division by zero

        # Step 3: Normalize CDF to [0,255]
        cdf_min = cdf_masked.min()
        cdf_max = cdf_masked.max()
        cdf_scaled = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)

        # Step 4: Fill masked values with 0 and map intensities
        cdf_final = np.ma.filled(cdf_scaled, 0).astype(np.uint8)
        img_equalized = cdf_final[img]

        return img_equalized

    def visualize_histogram_equalization(self, img: np.ndarray):
        """
        Compare original image & histogram with equalized version.
        """

        equalized_img = self.histogram_equalization(img)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # --- Original ---
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].hist(img.flatten(), bins=256, range=[0,256], color='gray')
        axes[0, 1].set_title('Original Histogram')
        axes[0, 1].set_xlim([0,255])

        # --- Equalized ---
        axes[1, 0].imshow(equalized_img, cmap='gray')
        axes[1, 0].set_title('Equalized Image')
        axes[1, 0].axis('off')

        axes[1, 1].hist(equalized_img.flatten(), bins=256, range=[0,256], color='blue')
        axes[1, 1].set_title('Equalized Histogram')
        axes[1, 1].set_xlim([0,255])

        plt.tight_layout()
        plt.show()

    def build_lookup_table(self, transform_func):
        """
        Build a lookup table (LUT) for all 256 possible pixel values (0–255)
        using a given transformation function.
        """
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = np.clip(transform_func(i), 0, 255)
        return lut

    def apply_lookup_table(self, img: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Apply a lookup table to the image.
        """
        return lut[img]

    def visualize_lut_gamma(self, img_array, gamma):
        """
        Example: Use LUT to perform gamma correction.
        """
        # Build LUT for gamma correction
        lut = self.build_lookup_table(lambda x: 255 * ((x / 255.0) ** gamma))
        transformed_img = self.apply_lookup_table(img_array, lut)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(transformed_img, cmap='gray')
        axes[0].set_title(f'LUT Gamma Correction (γ={gamma})')
        axes[0].axis('off')

        axes[1].hist(transformed_img.flatten(), bins=256, range=[0,256], color='blue')
        axes[1].set_title('Histogram (After LUT Transformation)')
        plt.tight_layout()
        plt.show()
