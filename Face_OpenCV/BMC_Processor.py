"""
BMC (Bilateral Median Convolution) Processor for Phase II
Enhances texture representation for robust face recognition
Combines bilateral filtering with median convolution for better occlusion handling
"""

import cv2
import numpy as np
from scipy import ndimage


class BMCProcessor:
    """
    Bilateral Median Convolution Processor
    Enhances LBP features by combining spatial, intensity, and median filtering
    """
    
    def __init__(self, kernel_size=5, sigma_space=75, sigma_color=75):
        """
        Initialize BMC processor
        
        Args:
            kernel_size: Size of the processing kernel (default: 5)
            sigma_space: Spatial sigma for bilateral filter (default: 75)
            sigma_color: Color/intensity sigma for bilateral filter (default: 75)
        """
        self.kernel_size = kernel_size
        self.sigma_space = sigma_space
        self.sigma_color = sigma_color
    
    def bilateral_median_convolution(self, image):
        """
        Apply Bilateral Median Convolution
        
        This combines:
        1. Bilateral filtering (edge-preserving smoothing)
        2. Median filtering (outlier removal)
        3. Weighted convolution based on spatial and intensity distance
        
        Args:
            image: Input grayscale image
            
        Returns:
            BMC processed image
        """
        h, w = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        pad = self.kernel_size // 2
        
        # Pad image
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        
        # Precompute spatial weights (Gaussian kernel)
        spatial_kernel = self._create_spatial_kernel()
        
        # Process each pixel
        for i in range(h):
            for j in range(w):
                # Extract neighborhood
                neighborhood = padded[i:i+self.kernel_size, j:j+self.kernel_size]
                center_val = image[i, j]
                
                # Compute intensity weights (bilateral component)
                intensity_diff = np.abs(neighborhood.astype(np.float32) - center_val)
                intensity_weights = np.exp(-(intensity_diff ** 2) / (2 * self.sigma_color ** 2))
                
                # Combine spatial and intensity weights
                combined_weights = spatial_kernel * intensity_weights
                
                # Normalize weights
                combined_weights = combined_weights / np.sum(combined_weights)
                
                # Apply median filtering to neighborhood first
                median_neighborhood = ndimage.median_filter(neighborhood, size=3)
                
                # Weighted sum using BMC weights
                result[i, j] = np.sum(median_neighborhood * combined_weights)
        
        return result.astype(np.uint8)
    
    def _create_spatial_kernel(self):
        """
        Create Gaussian spatial kernel for BMC
        
        Returns:
            Spatial weight kernel
        """
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        center = self.kernel_size // 2
        
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dist = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = np.exp(-dist / (2 * self.sigma_space ** 2))
        
        return kernel
    
    def fast_bmc(self, image, strength='light'):
        """
        Fast approximation of BMC using separable filters
        Suitable for real-time processing
        
        Args:
            image: Input grayscale image
            strength: 'light', 'medium', or 'heavy' (default: 'light' for compatibility)
            
        Returns:
            Fast BMC processed image
        """
        if strength == 'light':
            # Light processing - better for old training images
            img = cv2.bilateralFilter(image, 5, 50, 50)
            return img
        elif strength == 'medium':
            # Medium processing
            img = cv2.medianBlur(image, 3)
            img = cv2.bilateralFilter(img, 5, 60, 60)
            return img
        else:  # heavy
            # Original heavy processing
            img = cv2.medianBlur(image, 3)
            img = cv2.bilateralFilter(img, self.kernel_size, self.sigma_color, self.sigma_space)
            img = cv2.medianBlur(img, 3)
            return img
    
    def multi_scale_bmc(self, image, scales=[1.0, 0.75, 0.5]):
        """
        Apply BMC at multiple scales and combine
        Captures both fine and coarse texture patterns
        
        Args:
            image: Input grayscale image
            scales: List of scale factors (default: [1.0, 0.75, 0.5])
            
        Returns:
            Multi-scale BMC processed image
        """
        h, w = image.shape
        results = []
        
        for scale in scales:
            # Resize image
            new_size = (int(w * scale), int(h * scale))
            scaled_img = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Apply fast BMC
            bmc_img = self.fast_bmc(scaled_img)
            
            # Resize back to original size
            bmc_img = cv2.resize(bmc_img, (w, h), interpolation=cv2.INTER_LINEAR)
            results.append(bmc_img)
        
        # Combine scales (weighted average favoring original scale)
        weights = [0.6, 0.25, 0.15]  # Favor original scale
        combined = np.zeros_like(image, dtype=np.float32)
        
        for img, weight in zip(results, weights):
            combined += img.astype(np.float32) * weight
        
        return combined.astype(np.uint8)
    
    def adaptive_bmc(self, image):
        """
        Adaptive BMC that adjusts parameters based on local image characteristics
        
        Args:
            image: Input grayscale image
            
        Returns:
            Adaptive BMC processed image
        """
        # Calculate local variance to determine smoothness
        mean = cv2.blur(image, (5, 5))
        sqr_mean = cv2.blur(image ** 2, (5, 5))
        variance = sqr_mean - mean ** 2
        
        # Normalize variance to [0, 1]
        variance_norm = cv2.normalize(variance, None, 0, 1, cv2.NORM_MINMAX)
        
        # High variance regions (edges/textures) need less smoothing
        # Low variance regions (flat areas) can handle more smoothing
        
        # Create adaptive kernel
        h, w = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        
        # Divide into blocks for efficiency
        block_size = 16
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Extract block
                i_end = min(i + block_size, h)
                j_end = min(j + block_size, w)
                block = image[i:i_end, j:j_end]
                var_block = variance_norm[i:i_end, j:j_end]
                
                # Determine adaptive sigma based on average variance
                avg_var = np.mean(var_block)
                
                # Low variance -> more smoothing, high variance -> less smoothing
                adaptive_sigma = self.sigma_color * (1 - avg_var * 0.5)
                
                # Apply BMC with adaptive parameters
                if block.size > 0:
                    processed = cv2.bilateralFilter(
                        block, 
                        self.kernel_size, 
                        adaptive_sigma, 
                        self.sigma_space
                    )
                    processed = cv2.medianBlur(processed, 3)
                    result[i:i_end, j:j_end] = processed
        
        return result.astype(np.uint8)
    
    def enhance_for_lbph(self, image, mode='fast'):
        """
        Enhance image specifically for LBPH feature extraction
        
        Args:
            image: Input grayscale face image
            mode: 'fast' (real-time), 'standard', or 'advanced' (thorough)
            
        Returns:
            BMC enhanced image ready for LBPH
        """
        if mode == 'fast':
            # Fast BMC for real-time processing
            return self.fast_bmc(image)
        
        elif mode == 'standard':
            # Multi-scale BMC for better texture capture
            return self.multi_scale_bmc(image, scales=[1.0, 0.8])
        
        elif mode == 'advanced':
            # Full adaptive BMC for maximum robustness
            return self.adaptive_bmc(image)
        
        else:
            return image
    
    def compute_bmc_texture_descriptor(self, image):
        """
        Compute texture descriptor enhanced by BMC
        Useful for comparing texture quality
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary with texture metrics
        """
        # Apply BMC
        bmc_img = self.fast_bmc(image)
        
        # Compute texture metrics
        # 1. Entropy (texture complexity)
        hist = cv2.calcHist([bmc_img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # 2. Contrast
        mean_val = np.mean(bmc_img)
        contrast = np.sqrt(np.mean((bmc_img - mean_val) ** 2))
        
        # 3. Edge strength
        edges = cv2.Canny(bmc_img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'entropy': entropy,
            'contrast': contrast,
            'edge_density': edge_density,
            'mean_intensity': mean_val
        }


def test_bmc_processor():
    """Test BMC processing functions"""
    print("Testing BMC Processor Module...")
    
    # Create test image with some texture
    test_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    # Add some structure (simulate face features)
    cv2.circle(test_img, (100, 80), 20, 200, -1)  # Eye
    cv2.circle(test_img, (100, 150), 30, 180, -1)  # Nose
    
    bmc = BMCProcessor(kernel_size=5, sigma_space=75, sigma_color=75)
    
    # Test fast BMC
    print("Testing fast BMC...")
    fast_result = bmc.fast_bmc(test_img)
    
    # Test multi-scale BMC
    print("Testing multi-scale BMC...")
    multi_result = bmc.multi_scale_bmc(test_img)
    
    # Test adaptive BMC
    print("Testing adaptive BMC...")
    adaptive_result = bmc.adaptive_bmc(test_img)
    
    # Test texture descriptor
    print("Computing texture descriptors...")
    descriptor = bmc.compute_bmc_texture_descriptor(test_img)
    
    print("✅ All BMC functions working!")
    print(f"Input shape: {test_img.shape}")
    print(f"Output shape: {fast_result.shape}")
    print(f"Texture descriptor: {descriptor}")


if __name__ == "__main__":
    test_bmc_processor()
