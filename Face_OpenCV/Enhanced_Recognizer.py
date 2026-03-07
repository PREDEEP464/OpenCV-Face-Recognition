"""
Enhanced LBPH Recognizer with Optimized Parameters for Phase II
Includes dynamic threshold adjustment and performance optimization
"""

import cv2
import numpy as np


class EnhancedLBPHRecognizer:
    """
    Enhanced LBPH Face Recognizer with optimized parameters
    and adaptive threshold management
    """
    
    def __init__(self, radius=2, neighbors=16, grid_x=8, grid_y=8, threshold=70.0):
        """
        Initialize Enhanced LBPH Recognizer with optimized parameters
        
        Args:
            radius: Radius of LBP pattern (default: 2, Phase I used 1)
            neighbors: Number of neighbors for LBP (default: 16, Phase I used 8)
            grid_x: Number of cells in horizontal direction (default: 8)
            grid_y: Number of cells in vertical direction (default: 8)
            threshold: Recognition threshold (default: 70.0)
        
        Optimized Parameters Explanation:
        - Increased radius (1→2): Captures larger texture patterns
        - Increased neighbors (8→16): More detailed texture description
        - Grid 8x8: Better spatial information without overfitting
        """
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.threshold = threshold
        
        # Create recognizer with optimized parameters
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=self.radius,
            neighbors=self.neighbors,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            threshold=self.threshold
        )
        
        # Performance tracking
        self.prediction_count = 0
        self.total_confidence = 0
        self.recognition_history = []
    
    def train(self, faces, labels):
        """
        Train the recognizer with face data
        
        Args:
            faces: List of face images (numpy arrays)
            labels: List of corresponding labels (integers)
        """
        self.recognizer.train(faces, np.array(labels))
        print(f"✅ Enhanced LBPH trained with radius={self.radius}, neighbors={self.neighbors}")
    
    def predict(self, face):
        """
        Predict identity with enhanced confidence handling
        
        Args:
            face: Face image (numpy array)
            
        Returns:
            Tuple of (label, confidence, recognition_quality)
        """
        label, confidence = self.recognizer.predict(face)
        
        # Track prediction statistics
        self.prediction_count += 1
        self.total_confidence += confidence
        
        # Calculate recognition quality score (0-100, higher is better)
        quality = self._calculate_recognition_quality(confidence)
        
        # Store in history (keep last 100 predictions)
        self.recognition_history.append({
            'label': label,
            'confidence': confidence,
            'quality': quality
        })
        if len(self.recognition_history) > 100:
            self.recognition_history.pop(0)
        
        return label, confidence, quality
    
    def _calculate_recognition_quality(self, confidence):
        """
        Calculate recognition quality score from confidence
        Lower LBPH confidence = better match
        
        Args:
            confidence: Raw LBPH confidence (0-150 typical range)
            
        Returns:
            Quality score (0-100, higher is better)
        """
        # Map confidence to quality (inverse relationship)
        # 0-30: Excellent (90-100 quality)
        # 30-50: Good (70-89 quality)
        # 50-70: Fair (50-69 quality)
        # 70+: Poor (0-49 quality)
        
        if confidence < 30:
            quality = 100 - (confidence * 0.33)  # 90-100 range
        elif confidence < 50:
            quality = 90 - ((confidence - 30) * 1.0)  # 70-90 range
        elif confidence < 70:
            quality = 70 - ((confidence - 50) * 1.0)  # 50-70 range
        else:
            quality = max(0, 50 - ((confidence - 70) * 0.5))  # 0-50 range
        
        return max(0, min(100, quality))
    
    def get_adaptive_threshold(self, recent_window=10):
        """
        Calculate adaptive threshold based on recent predictions
        
        Args:
            recent_window: Number of recent predictions to consider
            
        Returns:
            Adaptive threshold value
        """
        if len(self.recognition_history) < recent_window:
            return self.threshold
        
        # Get recent confidences
        recent = self.recognition_history[-recent_window:]
        recent_confidences = [r['confidence'] for r in recent]
        
        # Calculate mean and std
        mean_conf = np.mean(recent_confidences)
        std_conf = np.std(recent_confidences)
        
        # Adaptive threshold: mean + 1.5*std
        adaptive_thresh = mean_conf + 1.5 * std_conf
        
        # Clamp to reasonable range [50, 100]
        return max(50, min(100, adaptive_thresh))
    
    def is_confident_match(self, confidence, use_adaptive=False):
        """
        Determine if prediction is confident enough
        
        Args:
            confidence: Prediction confidence
            use_adaptive: Use adaptive threshold instead of fixed
            
        Returns:
            Boolean indicating if match is confident
        """
        threshold = self.get_adaptive_threshold() if use_adaptive else self.threshold
        return confidence < threshold
    
    def get_statistics(self):
        """
        Get recognition statistics
        
        Returns:
            Dictionary with performance statistics
        """
        if self.prediction_count == 0:
            return {
                'total_predictions': 0,
                'avg_confidence': 0,
                'avg_quality': 0,
                'adaptive_threshold': self.threshold
            }
        
        avg_confidence = self.total_confidence / self.prediction_count
        avg_quality = np.mean([r['quality'] for r in self.recognition_history]) if self.recognition_history else 0
        
        return {
            'total_predictions': self.prediction_count,
            'avg_confidence': avg_confidence,
            'avg_quality': avg_quality,
            'adaptive_threshold': self.get_adaptive_threshold(),
            'recent_history_size': len(self.recognition_history)
        }
    
    def reset_statistics(self):
        """Reset performance tracking statistics"""
        self.prediction_count = 0
        self.total_confidence = 0
        self.recognition_history = []
    
    def update_parameters(self, radius=None, neighbors=None, threshold=None):
        """
        Update recognizer parameters (requires retraining)
        
        Args:
            radius: New radius value
            neighbors: New neighbors value
            threshold: New threshold value
        """
        if radius is not None:
            self.radius = radius
        if neighbors is not None:
            self.neighbors = neighbors
        if threshold is not None:
            self.threshold = threshold
        
        # Recreate recognizer with new parameters
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=self.radius,
            neighbors=self.neighbors,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            threshold=self.threshold
        )
        
        print(f"⚙️ Updated parameters: radius={self.radius}, neighbors={self.neighbors}, threshold={self.threshold}")


def create_enhanced_recognizer(radius=2, neighbors=16, grid_x=8, grid_y=8, threshold=70.0):
    """
    Factory function to create enhanced recognizer
    
    Args:
        radius: LBP radius
        neighbors: Number of LBP neighbors
        grid_x: Horizontal grid cells
        grid_y: Vertical grid cells
        threshold: Recognition threshold
        
    Returns:
        EnhancedLBPHRecognizer instance
    """
    return EnhancedLBPHRecognizer(radius, neighbors, grid_x, grid_y, threshold)


def compare_configurations():
    """
    Compare different LBPH configurations
    Useful for parameter tuning
    """
    print("\n" + "="*60)
    print("LBPH Configuration Comparison")
    print("="*60)
    
    configs = [
        {'name': 'Phase I (Original)', 'radius': 1, 'neighbors': 8, 'grid': 8},
        {'name': 'Phase II (Standard)', 'radius': 2, 'neighbors': 16, 'grid': 8},
        {'name': 'Phase II (High Detail)', 'radius': 3, 'neighbors': 24, 'grid': 10},
        {'name': 'Phase II (Speed)', 'radius': 1, 'neighbors': 8, 'grid': 6},
    ]
    
    for config in configs:
        feature_count = config['neighbors'] * config['grid'] ** 2
        print(f"\n{config['name']}:")
        print(f"  Radius: {config['radius']}, Neighbors: {config['neighbors']}, Grid: {config['grid']}x{config['grid']}")
        print(f"  Feature vector size: {feature_count}")
        print(f"  Expected performance: ", end="")
        
        if config['neighbors'] >= 16:
            print("High accuracy, moderate speed")
        else:
            print("Moderate accuracy, high speed")


if __name__ == "__main__":
    print("Testing Enhanced LBPH Recognizer...")
    
    # Create recognizer
    recognizer = create_enhanced_recognizer()
    
    # Test with dummy data
    dummy_faces = [np.random.randint(0, 255, (200, 200), dtype=np.uint8) for _ in range(3)]
    dummy_labels = [0, 1, 1]
    
    recognizer.train(dummy_faces, dummy_labels)
    
    # Test prediction
    test_face = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    label, conf, quality = recognizer.predict(test_face)
    
    print(f"✅ Prediction: Label={label}, Confidence={conf:.2f}, Quality={quality:.2f}")
    
    # Show statistics
    stats = recognizer.get_statistics()
    print(f"📊 Statistics: {stats}")
    
    # Compare configurations
    compare_configurations()
