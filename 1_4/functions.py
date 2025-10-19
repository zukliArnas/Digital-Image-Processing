from libtiff import TIFF  # type: ignore
import numpy as np # type: ignore
from numpy import ndarray # type: ignore
import math
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

    def index_to_local(self, i, j, H, W):
        x = j - (W - 1) / 2
        y = (H - 1) / 2 - i
        return x, y

    def local_to_world(self, x, y, pixel_size_x, pixel_size_y):
        X = x * pixel_size_x
        Y = y * pixel_size_y
        return X, Y

    def index_to_world(self, i, j, H, W, pixel_size_x, pixel_size_y):
        x, y = self.index_to_local(i, j, H, W)
        X, Y = self.local_to_world(x, y, pixel_size_x, pixel_size_y)
        return X, Y

    @staticmethod
    def visualize_coordinate_systems(H=256, W=256, pixel_size_x=0.5, pixel_size_y=0.5):
        # Create a blank image (for visualization)
        img = np.zeros((H, W), dtype=np.uint8)
        img[H//2, :] = 255  # horizontal center line
        img[:, W//2] = 255  # vertical center line

        # Define a few sample points: top-left, center, bottom-right
        sample_points = [(0, 0), (H//2, W//2), (H-1, W-1)]

        # --- Compute coordinate mappings ---
        coords = []
        for (i, j) in sample_points:
            # index -> local
            x = j - (W - 1) / 2
            y = (H - 1) / 2 - i

            # local -> world
            X = x * pixel_size_x
            Y = y * pixel_size_y

            coords.append(((i, j), (x, y), (X, Y)))

        # --- Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1️⃣ Index Coordinates
        axes[0].imshow(img, cmap='gray', origin='upper')
        for (i, j), _, _ in coords:
            axes[0].plot(j, i, 'ro')
            axes[0].text(j + 5, i - 5, f"({i},{j})", color='red')
        axes[0].set_title("Index Coordinates\n(origin top-left)")
        axes[0].invert_yaxis()  # image coords
        axes[0].set_xlabel("j (column)")
        axes[0].set_ylabel("i (row)")

        # 2️⃣ Local Coordinates
        xs = [x for _, (x, _), _ in coords]
        ys = [y for _, (_, y), _ in coords]
        axes[1].scatter(xs, ys, color='blue')
        for (i, j), (x, y), _ in coords:
            axes[1].text(x + 5, y + 5, f"({int(x)}, {int(y)})", color='blue')
        axes[1].set_title("Local Coordinates\n(origin centered)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].axhline(0, color='gray', linestyle='--')
        axes[1].axvline(0, color='gray', linestyle='--')
        axes[1].set_aspect('equal')

        # 3️⃣ World Coordinates
        Xs = [X for _, _, (X, _) in coords]
        Ys = [Y for _, _, (_, Y) in coords]
        axes[2].scatter(Xs, Ys, color='green')
        for (i, j), _, (X, Y) in coords:
            axes[2].text(X + 2, Y + 2, f"({X:.1f},{Y:.1f})", color='green')
        axes[2].set_title("World Coordinates\n(scaled by pixel size)")
        axes[2].set_xlabel("X [mm]")
        axes[2].set_ylabel("Y [mm]")
        axes[2].axhline(0, color='gray', linestyle='--')
        axes[2].axvline(0, color='gray', linestyle='--')
        axes[2].set_aspect('equal')

        plt.tight_layout()
        plt.show()

    def affine_translation(self, tx, ty):
        return np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)

    def affine_scaling(self, sx, sy):
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def affine_rotation(self, theta_deg):
        theta = math.radians(theta_deg)
        return np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def affine_shear(self, kx, ky):
        return np.array([
            [1, kx, 0],
            [ky, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def combine_affine(self, matrices):
        """Multiply affine matrices in given order."""
        result = np.eye(3, dtype=np.float32)
        for M in matrices:
            result = M @ result  # Right-multiplication
        return result

    def apply_affine_transform(self, image, M):
        """
        Applies an affine transformation to a grayscale image (without interpolation).
        Any unmapped pixels will appear black.
        """
        H, W = image.shape
        transformed = np.zeros_like(image)
        M_inv = np.linalg.inv(M)

        for i_out in range(H):
            for j_out in range(W):
                src_coords = M_inv @ np.array([j_out, i_out, 1])
                x_src, y_src = src_coords[0], src_coords[1]

                j_src, i_src = int(round(x_src)), int(round(y_src))
                if 0 <= i_src < H and 0 <= j_src < W:
                    transformed[i_out, j_out] = image[i_src, j_src]

        return transformed


    def visualize_composite_affine(self, image_array):
        """
        Apply a composite affine transformation to an image (translation, scaling, rotation)
        and visualize the result.
        """
        # Define transformation matrices
        R_centered = self.affine_rotation_around_center(45, image_array.shape)
        T = self.affine_translation(3, 3)
        S = self.affine_scaling(1.2, 1.2)
        M_composite = self.combine_affine([T, S, R_centered])
        print("Composite Affine Transformation Matrix:\n", M_composite)

        # Apply transformation
        transformed_img = self.apply_affine_transform(image_array, M_composite)

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(transformed_img, cmap='gray')
        axes[1].set_title('Transformed (No Interpolation)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    def get_pixel_value(self, img, x, y, method="nearest"):
        """Fetch interpolated pixel value from image at (x, y)."""
        H, W = img.shape

        if method == "nearest":
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W:
                return img[yi, xi]
            else:
                return 0

        elif method == "bilinear":
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = x0 + 1, y0 + 1

            if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
                return 0

            Ia = img[y0, x0]
            Ib = img[y0, x1]
            Ic = img[y1, x0]
            Id = img[y1, x1]

            wa = (x1 - x) * (y1 - y)
            wb = (x - x0) * (y1 - y)
            wc = (x1 - x) * (y - y0)
            wd = (x - x0) * (y - y0)

            return Ia * wa + Ib * wb + Ic * wc + Id * wd

        else:
            raise ValueError("Unknown interpolation method")

    def apply_affine_transform_interp(self, image, M, method="nearest"):
        """Apply affine transform using nearest or bilinear interpolation."""
        H, W = image.shape
        transformed = np.zeros((int(H * 1.5), int(W * 1.5)), dtype=np.uint8)
        M_inv = np.linalg.inv(M)

        for i_out in range(H):
            for j_out in range(W):
                src_coords = M_inv @ np.array([j_out, i_out, 1])
                x_src, y_src = src_coords[0], src_coords[1]
                transformed[i_out, j_out] = self.get_pixel_value(image, x_src, y_src, method)

        return np.clip(transformed, 0, 255).astype(np.uint8)


    def visualize_interpolation_effects(self, image_array):
        R_centered = self.affine_rotation_around_center(45, image_array.shape)
        T = self.affine_translation(3, 3)
        S = self.affine_scaling(1.2, 1.2)
        M_composite = self.combine_affine([T, S, R_centered])

        img_nearest = self.apply_affine_transform_interp(image_array, M_composite, "nearest")
        img_bilinear = self.apply_affine_transform_interp(image_array, M_composite, "bilinear")

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(img_nearest, cmap='gray')
        axes[1].set_title("Affine Transform – Nearest Neighbor")
        axes[1].axis("off")

        axes[2].imshow(img_bilinear, cmap='gray')
        axes[2].set_title("Affine Transform – Bilinear")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    def affine_rotation_around_center(self, angle_deg, img_shape):
            """Rotation matrix around image center."""
            h, w = img_shape
            cx, cy = w / 2, h / 2
            angle = np.deg2rad(angle_deg)

            # Translation to center, rotation, and translation back
            T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0, 0, 1]
            ])
            T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

            return T2 @ R @ T1
