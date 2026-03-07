"""
Enhanced Preprocessing Module for Phase II
Implements illumination normalization and noise reduction techniques
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """Advanced preprocessing pipeline for face images"""
    
    def __init__(self, clahe_clip_limit=2.0, clahe_tile_size=8):
        """
        Initialize preprocessor with CLAHE parameters
        
        Args:
            clahe_clip_limit: Threshold for contrast limiting (default: 2.0)
            clahe_tile_size: Size of grid for histogram equalization (default: 8)
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, 
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
    
    def apply_clahe(self, image):
        """
        Apply Contrast Limited Adaptive Histogram Equalization
        Improves contrast in images with varying illumination
        
        Args:
            image: Grayscale image
            
        Returns:
            CLAHE enhanced image
        """
        return self.clahe.apply(image)
    
    def gamma_correction(self, image, gamma=1.2):
        """
        Apply gamma correction for exposure adjustment
        
        Args:
            image: Input grayscale image
            gamma: Gamma value (>1 brightens, <1 darkens)
            
        Returns:
            Gamma corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter for edge-preserving smoothing
        Reduces noise while keeping sharp edges
        
        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def median_filter(self, image, kernel_size=5):
        """
        Apply median filter for salt-and-pepper noise removal
        
        Args:
            image: Input image
            kernel_size: Size of the median filter kernel (must be odd)
            
        Returns:
            Median filtered image
        """
        return cv2.medianBlur(image, kernel_size)
    
    def gaussian_filter(self, image, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian filter for general noise reduction
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation of Gaussian kernel
            
        Returns:
            Gaussian filtered image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def homomorphic_filter(self, image, d0=10, gamma_l=0.3, gamma_h=1.5):
        """
        Apply homomorphic filtering for illumination normalization
        Reduces lighting variations while enhancing details
        
        Args:
            image: Input grayscale image
            d0: Cutoff frequency
            gamma_l: Low frequency gain
            gamma_h: High frequency gain
            
        Returns:
            Homomorphic filtered image
        """
        # Convert to float and take log
        image_float = np.float32(image) + 1.0
        image_log = np.log(image_float)
        
        # FFT
        dft = cv2.dft(image_log, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create Gaussian high-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create filter
        mask = np.zeros((rows, cols, 2), np.float32)
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow)**2 + (j - ccol)**2)
                h = (gamma_h - gamma_l) * (1 - np.exp(-(d**2) / (2 * (d0**2)))) + gamma_l
                mask[i, j] = h
        
        # Apply filter
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        # Exponential and normalize
        img_back = np.exp(img_back)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        
        return np.uint8(img_back)
    
    def preprocess_full_pipeline(self, image, mode='standard'):
        """
        Apply complete preprocessing pipeline
        
        Args:
            image: Input grayscale image
            mode: 'standard' (fast) or 'advanced' (thorough)
            
        Returns:
            Fully preprocessed image
        """
        if mode == 'standard':
            # Standard pipeline: CLAHE + Bilateral filter
            img = self.apply_clahe(image)
            img = self.bilateral_filter(img, d=5, sigma_color=50, sigma_space=50)
            return img
        
        elif mode == 'advanced':
            # Advanced pipeline: All techniques
            # 1. Noise reduction
            img = self.median_filter(image, kernel_size=3)
            
            # 2. Illumination normalization
            img = self.homomorphic_filter(img, d0=10, gamma_l=0.4, gamma_h=1.8)
            
            # 3. CLAHE for local contrast
            img = self.apply_clahe(img)
            
            # 4. Edge-preserving smoothing
            img = self.bilateral_filter(img, d=7, sigma_color=60, sigma_space=60)
            
            return img
        
        else:
            # No preprocessing
            return image
    
    def preprocess_training_image(self, image):
        """
        Preprocessing specifically optimized for training data
        More aggressive to handle diverse conditions
        
        Args:
            image: Training face image (grayscale)
            
        Returns:
            Preprocessed training image
        """
        # Apply comprehensive preprocessing for training
        img = self.median_filter(image, kernel_size=3)
        img = self.apply_clahe(img)
        img = self.bilateral_filter(img, d=7, sigma_color=60, sigma_space=60)
        return img
    
    def preprocess_realtime_image(self, image):
        """
        Preprocessing optimized for real-time processing
        Faster with minimal quality loss
        
        Args:
            image: Real-time face image (grayscale)
            
        Returns:
            Preprocessed real-time image
        """
        # Fast preprocessing for real-time
        img = self.apply_clahe(image)
        img = self.bilateral_filter(img, d=5, sigma_color=50, sigma_space=50)
        return img


def test_preprocessing():
    """Test preprocessing functions"""
    print("Testing Preprocessing Module...")
    
    # Create test image
    test_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor()
    
    # Test each function
    clahe_img = preprocessor.apply_clahe(test_img)
    gamma_img = preprocessor.gamma_correction(test_img)
    bilateral_img = preprocessor.bilateral_filter(test_img)
    median_img = preprocessor.median_filter(test_img)
    gaussian_img = preprocessor.gaussian_filter(test_img)
    
    # Test pipelines
    standard_img = preprocessor.preprocess_full_pipeline(test_img, 'standard')
    advanced_img = preprocessor.preprocess_full_pipeline(test_img, 'advanced')
    
    print("✅ All preprocessing functions working!")
    print(f"Input shape: {test_img.shape}")
    print(f"Output shape: {standard_img.shape}")


if __name__ == "__main__":
    test_preprocessing()
