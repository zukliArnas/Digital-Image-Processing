from libtiff import TIFF  # type: ignore
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

    def to_float_img(self, img_uint8: np.ndarray) -> np.ndarray:
        """Convert 8-bit uint image to float (range [0,1])."""
        return img_uint8.astype(np.float32) / 255.0

    def to_uint8_img(self, img_float: np.ndarray) -> np.ndarray:
        """Convert float image [0,1] to 8-bit uint.""" 
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

        filtered = np.zeros_like(img_float)

        # Convolution
        for i in range(height):
            for j in range(width):
                region = padded_img[i:i+k_h, j:j+k_w]
                filtered[i, j] = np.sum(region * kernel)

        return np.clip(filtered, 0.0, 1.0)

    def visualize_blur(self, img_array: np.ndarray) -> None:
        """ Demonstrate spatial blurring using a mean filter."""
        img_float = self.to_float_img(img_array)

        # Define a simple 3x3 averaging kernel
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0

        # Apply filter
        blurred = self.apply_spatial_filter(img_float, kernel)
        blurred_uint8 = self.to_uint8_img(blurred)

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

    def apply_filter(self, img_float: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Basic convolution filter for float images."""
        h, w = img_float.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        padded = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        output = np.zeros_like(img_float)

        for i in range(h):
            for j in range(w):
                region = padded[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(region * kernel)

        return output

    def visualize_gradients_and_laplacian(self, img_array):
        """Compute and visualize partial derivatives and Laplacian."""
        img_float = self.to_float_img(img_array)

        #  Sobel derivative filters 
        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Gy = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

        grad_x = self.apply_filter(img_float, Gx)
        grad_y = self.apply_filter(img_float, Gy)

        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = grad_mag / grad_mag.max()  # normalize to [0,1]

        Laplacian = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]], dtype=np.float32)
        laplace = self.apply_filter(img_float, Laplacian)
        laplace = (laplace - laplace.min()) / (laplace.max() - laplace.min())

        #  Convert to 8-bit for display 
        grad_disp = self.to_uint8_img(grad_mag)
        lap_disp = self.to_uint8_img(laplace)

        _, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(img_array, cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(grad_disp, cmap='gray')
        axes[0, 1].set_title("Gradient Magnitude (Edges)")
        axes[0, 1].axis('off')

        axes[1, 0].imshow(grad_x, cmap='gray')
        axes[1, 0].set_title("Gradient X")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(lap_disp, cmap='gray')
        axes[1, 1].set_title("Laplacian (2nd Derivative)")
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def unsharp_mask(self, img_float: np.ndarray, kernel_size: int = 5, amount: float = 1.0) -> np.ndarray:
        """Perform image sharpening using Unsharp Mask."""
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        blurred = self.apply_spatial_filter(img_float, kernel)
        mask = img_float - blurred  # edge information
        sharpened = img_float + amount * mask
        return np.clip(sharpened, 0.0, 1.0)

    def laplacian_sharpen(self, img_float: np.ndarray, c: float = 1.0) -> np.ndarray:
        """Perform image sharpening using Laplacian operator."""
        laplacian_kernel = np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]], dtype=np.float32)
        laplace = self.apply_filter(img_float, laplacian_kernel)
        sharpened = img_float - c * laplace
        return np.clip(sharpened, 0.0, 1.0)

    def visualize_sharpening(self, img_array: np.ndarray):
        """Compare unsharp mask and Laplacian sharpening visually."""
        img_float = self.to_float_img(img_array)

        unsharp = self.unsharp_mask(img_float, kernel_size=5, amount=1.0)
        laplace_sharp = self.laplacian_sharpen(img_float, c=1.0)

        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(self.to_uint8_img(unsharp), cmap='gray')
        axes[1].set_title("Unsharp Mask Sharpening")
        axes[1].axis('off')

        axes[2].imshow(self.to_uint8_img(laplace_sharp), cmap='gray')
        axes[2].set_title("Laplacian Sharpening")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    def multistep_example_3_43(self, img_array) -> None:

        img = self.to_float_img(img_array)

        # Step a: Original
        a = img

        # Step b: Laplacian
        laplacian_kernel = np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]], dtype=np.float32)
        b = self.apply_filter(img, laplacian_kernel)

        # Step c: Sharpening (a + b)
        c = np.clip(a + b, 0, 1)

        # Step d: Sobel gradient magnitude
        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Gy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
        grad_x = self.apply_filter(a, Gx)
        grad_y = self.apply_filter(a, Gy)
        d = np.sqrt(grad_x**2 + grad_y**2)
        d /= d.max()  # normalize

        # Step e: Smooth d (use simple averaging kernel)
        smooth_kernel = np.ones((5, 5), dtype=np.float32) / 25
        e = self.apply_spatial_filter(d, smooth_kernel)

        # Step f: Mask (c * e)
        f = np.clip(c * e, 0, 1)

        # Step g: Sharpen again (a + f)
        g = np.clip(a + f, 0, 1)

        # Step h: Power-law transform
        gamma = 0.5  # brightens the image
        h = np.power(g, gamma)

        # Display all
        images = [a, b, c, d, e, f, g, h]
        titles = ["a) Original", "b) Laplacian", "c) a+b (Sharpened)",
                  "d) Sobel Gradient", "e) Smoothed Gradient", "f) c×e (Mask)",
                  "g) a+f (Sharpened Again)", "h) Power-law (γ=0.5)"]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for i, (img_step, title) in enumerate(zip(images, titles)):
            axes[i].imshow(self.to_uint8_img(img_step), cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
