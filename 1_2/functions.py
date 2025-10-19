from libtiff import TIFF # type: ignore
import numpy as np # type: ignore
from numpy import ndarray # type: ignore
import matplotlib.pyplot as plt # type: ignore

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
        """
        Read image data from the TIFF file, ensuring it’s 8-bit grayscale.
        """
        if not self.tif:
            return None
        
        try:
            arr_orig = self.tif.read_image()
                        
            # Check for 16-bit data and stop (as requested)
            if arr_orig.dtype == np.uint16:
                print(f"Error: Image {self.filename} is 16-bit ({arr_orig.dtype}).")
                return None
            
            # Handle Multi-channel images
            if arr_orig.ndim > 2:
                print(f"Input image is multi-channel ({arr_orig.ndim} dimensions). Taking the first channel as grayscale.")
                arr_orig = arr_orig[:, :, 0]
            
            # Ensure type is 8-bit unsigned integer
            if arr_orig.dtype != np.uint8:
                print(f"Converting input image from {arr_orig.dtype} to 8-bit (uint8).")
                arr_orig = arr_orig.astype(np.uint8)
                    
            return arr_orig

        except Exception as e:
            print(f"Error reading image data from {self.filename}: {e}")
            return None

    def power_law_transform(self, image_array: ndarray, gamma: float) -> ndarray:
        """
        Apply power-law intensity transformation.
        """
        # Normalize
        normalized_image = image_array / 255.0
        
        # Apply Power Law Transformation
        transformed_normalized = np.power(normalized_image, gamma)

        # Convert float [0, 1] back to 8-bit [0, 255]
        transformed_8bit = np.clip(transformed_normalized * 255, 0, 255).astype(np.uint8)
        return transformed_8bit

    def visualize_gamma_effect(self, gamma_values: list[float]) -> None:
        """
        Visualize the effect of multiple gamma corrections on an image.
        """
        original_img_array = self.read_image_data()

        if original_img_array is None:
            print("Visualization aborted because the image is not suitable.")
            return

        num_plots = len(gamma_values) + 1
        _, axes = plt.subplots(num_plots, 2, figsize=(10, 4 * num_plots))

        # Original image
        axes[0, 0].imshow(original_img_array, cmap='gray')
        axes[0, 0].set_title(f'Original Image ({self.filename})')
        axes[0, 0].axis('off')
        
        axes[0, 1].hist(original_img_array.flatten(), bins=256, range=[0, 256], color='gray')
        axes[0, 1].set_title('Original Histogram')
        axes[0, 1].set_xlim([0, 255])

        for i, gamma in enumerate(gamma_values, 1):
            arr_transformed = self.power_law_transform(original_img_array, gamma)
            
            # Image Plot
            axes[i, 0].imshow(arr_transformed, cmap='gray')
            axes[i, 0].set_title(rf'Transformed Image ($\gamma={gamma}$)')
            axes[i, 0].axis('off')
            
            axes[i, 1].hist(arr_transformed.flatten(), bins=256, range=[0, 256], color='blue')
            axes[i, 0].set_title(rf'Transformed Image ($\gamma={gamma}$)')
            axes[i, 1].set_xlim([0, 255])

        plt.tight_layout(pad=1.5, w_pad=2.0)
        plt.show()

    def transform_piece_wise(self, img: ndarray, r1: int, s1: int, r2: int, s2: int) -> ndarray:

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
    
    def visualize_piecewise_linear(self, img_array: ndarray, params: tuple[int, int, int, int] | None = None) -> None:
        """
        Visualize piecewise linear contrast stretching.
        Automatically estimates r1/r2 if not provided.
        """

        #  Automatic range detection 
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

        #  Apply transform 
        stretched_img = self.transform_piece_wise(img_array, r1, s1, r2, s2)
        print("Image stats  -> Min:", img_array.min(), " Max:", img_array.max())

        #  Plot results 
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(stretched_img, cmap='gray')
        axes[0].set_title(f'Piecewise Linear Transform\n(r1,s1,r2,s2)=({r1},{s1},{r2},{s2})')
        axes[0].axis('off')

        axes[1].hist(stretched_img.flatten(), bins=256, range=[0, 256], color='blue')
        axes[1].set_title('Histogram (After Stretching)')
        axes[1].set_xlim([0,255])
        plt.tight_layout()
        plt.show()

    def threshold(self, img: ndarray, thresh: int) -> ndarray:
        return np.uint8((img > thresh) * 255)

    def visualize_threshold(self, img_array, threshold_value) -> None:
        """Visualize thresholded image and its histogram."""
        thresholded_img = self.threshold(img_array, threshold_value)

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(thresholded_img, cmap='gray')
        axes[0].set_title(f'Thresholded Image (T={threshold_value})')
        axes[0].axis('off')

        axes[1].hist(thresholded_img.flatten(), bins=256, range=[0, 256], color='red')
        axes[1].set_title('Histogram (After Thresholding)')
        plt.show()

    def compute_histogram(self, img: ndarray) -> ndarray:
        """Compute the histogram (frequency of each intensity) for a grayscale image."""
        hist = np.zeros(256, dtype=int)
        for value in img.flatten():
            hist[value] += 1
        return hist

    def visualize_histogram(self, img: ndarray) -> None:
        """Display histogram plot"""
        hist = self.compute_histogram(img)
        plt.figure(figsize=(10,5))
        plt.bar(range(256), hist, color='gray')
        plt.title("Image Histogram", fontsize=18)
        plt.xlabel("Intensity Value", fontsize=14)
        plt.ylabel("Pixel Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim([0,255])
        plt.tight_layout()
        plt.show()

    def histogram_equalization(self, img: ndarray) -> ndarray:
        """
        Perform histogram equalization on a grayscale image.
        """
        hist, _ = np.histogram(img.flatten(), bins=256, range=[0,256])

        # Compute CDF 
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)  # mask zeros to avoid division by zero

        #Normalize CDF to [0,255]
        cdf_min = cdf_masked.min()
        cdf_max = cdf_masked.max()
        cdf_scaled = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)

        # Fill masked values with 0
        cdf_final = np.ma.filled(cdf_scaled, 0).astype(np.uint8)
        img_equalized = cdf_final[img]

        return img_equalized

    def visualize_histogram_equalization(self, img: ndarray) -> None:
        """
        Compare original image & histogram with equalized version.
        """

        equalized_img = self.histogram_equalization(img)

        _, axes = plt.subplots(2, 2, figsize=(10, 8))

        #  Original 
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].hist(img.flatten(), bins=256, range=[0,256], color='gray')
        axes[0, 1].set_title('Original Histogram')
        axes[0, 1].set_xlim([0,255])

        #  Equalized 
        axes[1, 0].imshow(equalized_img, cmap='gray')
        axes[1, 0].set_title('Equalized Image')
        axes[1, 0].axis('off')

        axes[1, 1].hist(equalized_img.flatten(), bins=256, range=[0,256], color='blue')
        axes[1, 1].set_title('Equalized Histogram')
        axes[1, 1].set_xlim([0,255])

        plt.tight_layout()
        plt.show()

    def build_lookup_table(self, transform_func: callable) -> ndarray:
        """
        Build a lookup table (LUT) using a given transformation function.
        """
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = np.clip(transform_func(i), 0, 255)
        return lut

    def apply_lookup_table(self, img: ndarray, lut: ndarray) -> ndarray:
        return lut[img]

    def visualize_lut_gamma(self, img_array: ndarray, gamma: float) -> None:
        """Perform gamma correction using a lookup table and visualize results."""
        lut = self.build_lookup_table(lambda x: 255 * ((x / 255.0) ** gamma))
        transformed_img = self.apply_lookup_table(img_array, lut)

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(transformed_img, cmap='gray')
        axes[0].set_title(f'LUT Gamma Correction (γ={gamma})')
        axes[0].axis('off')

        axes[1].hist(transformed_img.flatten(), bins=256, range=[0,256], color='blue')
        axes[1].set_title('Histogram (After LUT Transformation)')
        plt.tight_layout()
        plt.show()
