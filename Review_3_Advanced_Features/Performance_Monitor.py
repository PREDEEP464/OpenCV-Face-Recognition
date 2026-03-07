"""
Performance Monitor Module for Phase II
Tracks FPS, latency, accuracy metrics, and system performance
"""

import time
import cv2
import numpy as np
from collections import deque


class PerformanceMonitor:
    """
    Real-time performance monitoring for face recognition system
    """
    
    def __init__(self, window_size=30):
        """
        Initialize performance monitor
        
        Args:
            window_size: Size of rolling window for metrics (default: 30 frames)
        """
        self.window_size = window_size
        
        # FPS tracking
        self.frame_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.session_start = time.time()
        
        # Latency tracking
        self.detection_times = deque(maxlen=window_size)
        self.recognition_times = deque(maxlen=window_size)
        self.preprocessing_times = deque(maxlen=window_size)
        self.total_pipeline_times = deque(maxlen=window_size)
        
        # Recognition metrics
        self.recognition_count = 0
        self.detection_count = 0
        self.confidence_scores = deque(maxlen=window_size)
        
        # Temporary timers
        self._current_frame_start = None
        self._stage_start = None
    
    def start_frame(self):
        """Mark the start of a new frame"""
        self._current_frame_start = time.time()
        self.frame_count += 1
    
    def end_frame(self):
        """Mark the end of frame processing"""
        if self._current_frame_start is not None:
            frame_time = time.time() - self._current_frame_start
            self.frame_times.append(frame_time)
            self._current_frame_start = None
    
    def start_stage(self):
        """Mark start of a processing stage"""
        self._stage_start = time.time()
    
    def end_preprocessing(self):
        """Record preprocessing time"""
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            self.preprocessing_times.append(elapsed)
            self._stage_start = None
    
    def end_detection(self):
        """Record face detection time"""
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            self.detection_times.append(elapsed)
            self._stage_start = None
    
    def end_recognition(self):
        """Record recognition time"""
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            self.recognition_times.append(elapsed)
            self._stage_start = None
    
    def record_detection(self):
        """Increment detection count"""
        self.detection_count += 1
    
    def record_recognition(self, confidence):
        """
        Record recognition attempt
        
        Args:
            confidence: Confidence score from recognizer
        """
        self.recognition_count += 1
        self.confidence_scores.append(confidence)
    
    def get_fps(self):
        """
        Calculate current FPS
        
        Returns:
            Current FPS (frames per second)
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        # FPS = 1 / avg_frame_time
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_avg_latency(self, stage='total'):
        """
        Get average latency for a processing stage
        
        Args:
            stage: 'preprocessing', 'detection', 'recognition', or 'total'
            
        Returns:
            Average latency in milliseconds
        """
        if stage == 'preprocessing' and len(self.preprocessing_times) > 0:
            return np.mean(self.preprocessing_times) * 1000
        elif stage == 'detection' and len(self.detection_times) > 0:
            return np.mean(self.detection_times) * 1000
        elif stage == 'recognition' and len(self.recognition_times) > 0:
            return np.mean(self.recognition_times) * 1000
        elif stage == 'total' and len(self.frame_times) > 0:
            return np.mean(self.frame_times) * 1000
        return 0.0
    
    def get_avg_confidence(self):
        """
        Get average confidence score
        
        Returns:
            Average confidence (lower is better in LBPH)
        """
        if len(self.confidence_scores) == 0:
            return 0.0
        return np.mean(self.confidence_scores)
    
    def get_detection_rate(self):
        """
        Calculate face detection rate
        
        Returns:
            Percentage of frames with detected faces
        """
        if self.frame_count == 0:
            return 0.0
        return (self.detection_count / self.frame_count) * 100
    
    def get_recognition_rate(self):
        """
        Calculate recognition success rate
        
        Returns:
            Percentage of detections that resulted in recognition
        """
        if self.detection_count == 0:
            return 0.0
        return (self.recognition_count / self.detection_count) * 100
    
    def get_session_duration(self):
        """
        Get session duration in seconds
        
        Returns:
            Duration since session start
        """
        return time.time() - self.session_start
    
    def get_comprehensive_stats(self):
        """
        Get all performance statistics
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'fps': self.get_fps(),
            'avg_frame_time_ms': self.get_avg_latency('total'),
            'preprocessing_ms': self.get_avg_latency('preprocessing'),
            'detection_ms': self.get_avg_latency('detection'),
            'recognition_ms': self.get_avg_latency('recognition'),
            'avg_confidence': self.get_avg_confidence(),
            'detection_rate': self.get_detection_rate(),
            'recognition_rate': self.get_recognition_rate(),
            'total_frames': self.frame_count,
            'total_detections': self.detection_count,
            'total_recognitions': self.recognition_count,
            'session_duration': self.get_session_duration()
        }
    
    def render_stats_overlay(self, frame, mode='compact'):
        """
        Render performance statistics on frame
        
        Args:
            frame: Input frame (BGR)
            mode: 'compact' or 'detailed'
            
        Returns:
            Frame with statistics overlay
        """
        stats = self.get_comprehensive_stats()
        
        # Colors
        bg_color = (0, 0, 0)
        text_color = (0, 255, 0)
        
        if mode == 'compact':
            # Compact overlay - top right
            info_text = [
                f"FPS: {stats['fps']:.1f}",
                f"Latency: {stats['avg_frame_time_ms']:.1f}ms",
                f"Detections: {stats['total_detections']}"
            ]
            
            x, y = frame.shape[1] - 200, 30
            for i, text in enumerate(info_text):
                y_pos = y + i * 25
                # Background
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x-5, y_pos-text_h-5), (x+text_w+5, y_pos+5), bg_color, -1)
                # Text
                cv2.putText(frame, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        elif mode == 'detailed':
            # Detailed overlay - left side
            info_text = [
                "=== PERFORMANCE ===",
                f"FPS: {stats['fps']:.1f}",
                f"Frame Time: {stats['avg_frame_time_ms']:.1f}ms",
                f"Preprocessing: {stats['preprocessing_ms']:.1f}ms",
                f"Detection: {stats['detection_ms']:.1f}ms",
                f"Recognition: {stats['recognition_ms']:.1f}ms",
                "",
                "=== ACCURACY ===",
                f"Avg Confidence: {stats['avg_confidence']:.1f}",
                f"Detection Rate: {stats['detection_rate']:.1f}%",
                f"Recognition Rate: {stats['recognition_rate']:.1f}%",
                "",
                f"Total Frames: {stats['total_frames']}",
                f"Session: {stats['session_duration']:.0f}s"
            ]
            
            x, y = 10, 30
            for i, text in enumerate(info_text):
                y_pos = y + i * 22
                if text:  # Skip empty lines for background
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x-3, y_pos-text_h-3), (x+text_w+3, y_pos+3), bg_color, -1)
                    cv2.putText(frame, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        return frame
    
    def print_summary(self):
        """Print performance summary to console"""
        stats = self.get_comprehensive_stats()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        print(f"⚡ FPS: {stats['fps']:.2f}")
        print(f"⏱️  Avg Frame Time: {stats['avg_frame_time_ms']:.2f}ms")
        print(f"   └─ Preprocessing: {stats['preprocessing_ms']:.2f}ms")
        print(f"   └─ Detection: {stats['detection_ms']:.2f}ms")
        print(f"   └─ Recognition: {stats['recognition_ms']:.2f}ms")
        print()
        print(f"🎯 Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"👁️  Detection Rate: {stats['detection_rate']:.1f}%")
        print(f"✅ Recognition Rate: {stats['recognition_rate']:.1f}%")
        print()
        print(f"📈 Total Frames: {stats['total_frames']}")
        print(f"👥 Total Detections: {stats['total_detections']}")
        print(f"🔍 Total Recognitions: {stats['total_recognitions']}")
        print(f"⏰ Session Duration: {stats['session_duration']:.1f}s")
        print("="*60)
    
    def export_metrics(self, filename='performance_metrics.txt'):
        """
        Export metrics to file
        
        Args:
            filename: Output filename
        """
        stats = self.get_comprehensive_stats()
        
        with open(filename, 'w') as f:
            f.write("Face Recognition System - Performance Metrics\n")
            f.write("="*50 + "\n\n")
            
            f.write("Performance:\n")
            f.write(f"  FPS: {stats['fps']:.2f}\n")
            f.write(f"  Avg Frame Time: {stats['avg_frame_time_ms']:.2f}ms\n")
            f.write(f"  Preprocessing Time: {stats['preprocessing_ms']:.2f}ms\n")
            f.write(f"  Detection Time: {stats['detection_ms']:.2f}ms\n")
            f.write(f"  Recognition Time: {stats['recognition_ms']:.2f}ms\n\n")
            
            f.write("Accuracy:\n")
            f.write(f"  Avg Confidence: {stats['avg_confidence']:.2f}\n")
            f.write(f"  Detection Rate: {stats['detection_rate']:.1f}%\n")
            f.write(f"  Recognition Rate: {stats['recognition_rate']:.1f}%\n\n")
            
            f.write("Session:\n")
            f.write(f"  Total Frames: {stats['total_frames']}\n")
            f.write(f"  Total Detections: {stats['total_detections']}\n")
            f.write(f"  Total Recognitions: {stats['total_recognitions']}\n")
            f.write(f"  Duration: {stats['session_duration']:.1f}s\n")
        
        print(f"✅ Metrics exported to {filename}")
    
    def reset(self):
        """Reset all statistics"""
        self.frame_times.clear()
        self.detection_times.clear()
        self.recognition_times.clear()
        self.preprocessing_times.clear()
        self.total_pipeline_times.clear()
        self.confidence_scores.clear()
        
        self.frame_count = 0
        self.detection_count = 0
        self.recognition_count = 0
        self.session_start = time.time()


if __name__ == "__main__":
    print("Testing Performance Monitor...")
    
    monitor = PerformanceMonitor()
    
    # Simulate processing
    for i in range(10):
        monitor.start_frame()
        time.sleep(0.033)  # ~30 FPS
        
        monitor.start_stage()
        time.sleep(0.005)
        monitor.end_preprocessing()
        
        monitor.start_stage()
        time.sleep(0.010)
        monitor.end_detection()
        monitor.record_detection()
        
        monitor.start_stage()
        time.sleep(0.008)
        monitor.end_recognition()
        monitor.record_recognition(45.2)
        
        monitor.end_frame()
    
    # Print summary
    monitor.print_summary()
    
    print("\n✅ Performance Monitor test complete!")
