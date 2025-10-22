import cv2
import time
import numpy as np
import math
from datetime import datetime

# Enhanced UI colors with more vibrant palette
COLORS = {
    'green': (0, 255, 0),      # Recognized faces
    'red': (0, 0, 255),        # Unknown faces
    'blue': (255, 140, 0),     # UI elements - Deep Sky Blue
    'white': (255, 255, 255),  # Text
    'yellow': (0, 255, 255),   # Highlights - Cyan
    'purple': (255, 0, 255),   # Magenta for special effects
    'orange': (0, 165, 255),   # Orange for warnings
    'dark_gray': (40, 40, 40), # Background
    'light_blue': (255, 200, 100), # Light blue for accents
}

WINDOW_NAME = '🎯 Advanced Face Recognition System'

# Global animation variables
animation_frame = 0
scanning_animation = 0

def print_fancy_header():
    """Print an enhanced terminal header with emojis and styling"""
    print("\n" + "="*80)
    print("🎯 " + " "*20 + "ADVANCED FACE RECOGNITION SYSTEM" + " "*20 + " 🎯")
    print("="*80)
    print("🚀 Initializing AI-Powered Face Recognition Technology...")
    print("🔬 Powered by OpenCV & Deep Learning Algorithms")
    print("📅 Session Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)

def print_training_progress(current, total, name):
    """Print training progress with visual progress bar"""
    percentage = (current / total) * 100
    filled_length = int(50 * current // total)
    bar = "█" * filled_length + "░" * (50 - filled_length)
    print(f"\r🧠 Training Progress: |{bar}| {percentage:.1f}% - Processing {name}", end="", flush=True)

def print_session_stats(session_stats):
    """Print enhanced session statistics"""
    session_time = time.time() - session_stats['session_start']
    print("\n")
    print("📊 " + "="*25 + " SESSION SUMMARY " + "="*25 + " 📊")
    print(f"⏰ Session Duration: {session_time:.1f} seconds")
    if session_stats['detected_names']:
        print(f"👥 People Detected: {', '.join(sorted(session_stats['detected_names']))}")
    else:
        print("🔍 No people were detected during this session")
    print("="*78)

def make_fullscreen():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def show_loader(frame, message, progress=0.5, color=(100, 255, 100)):
    """Show a loading animation overlay"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    # Calculate center
    center_x, center_y = width // 2, height // 2
    
    # Draw loading circle
    radius = 60
    thickness = 8
    
    # Background circle
    cv2.circle(frame, (center_x, center_y), radius, (50, 50, 50), thickness)
    
    # Progress arc
    start_angle = -90  # Start from top
    end_angle = start_angle + (360 * progress)
    
    # Convert to OpenCV format (0-360 becomes 0-360)
    start_angle_cv = int(start_angle)
    end_angle_cv = int(end_angle)
    
    # Draw progress arc
    cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, start_angle_cv, end_angle_cv, color, thickness)
    
    # Loading text
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + radius + 50
    
    cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Percentage text
    percentage_text = f"{int(progress * 100)}%"
    perc_size = cv2.getTextSize(percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    perc_x = center_x - perc_size[0] // 2
    perc_y = center_y + 10
    
    cv2.putText(frame, percentage_text, (perc_x, perc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['white'], 2)
    
    return frame

def animated_loader(video_capture, message, duration=5.0, color=(100, 255, 100)):
    """Show animated loader for specified duration"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Calculate progress (0 to 1)
        elapsed = time.time() - start_time
        progress = min(elapsed / duration, 1.0)
        
        frame = show_loader(frame, message, progress, color)
        
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(30)  # ~30 FPS
    
    return True

def create_scanning_animation(frame, center_x, center_y, radius):
    """Create a scanning radar-like animation"""
    global scanning_animation
    
    # Increment animation frame
    scanning_animation += 0.1
    
    # Draw concentric circles for radar effect
    for i in range(3):
        circle_radius = int(radius + i * 20 + (scanning_animation * 10) % 60)
        alpha = max(0, 1 - (scanning_animation * 0.1) % 1)
        color = tuple(int(c * alpha) for c in COLORS['light_blue'])
        cv2.circle(frame, (center_x, center_y), circle_radius, color, 2)
    
    # Draw rotating scanning line
    angle = scanning_animation % (2 * math.pi)
    end_x = int(center_x + radius * math.cos(angle))
    end_y = int(center_y + radius * math.sin(angle))
    cv2.line(frame, (center_x, center_y), (end_x, end_y), COLORS['yellow'], 3)
    
    return frame

def add_enhanced_ui(frame, detected_count, recognized_count, total_trained, session_stats, system_paused):
    """Add enhanced UI with animations and better styling"""
    global animation_frame
    height, width = frame.shape[:2]
    animation_frame += 1
    
    # Animated top bar with gradient effect
    for i in range(70):
        alpha = (math.sin(animation_frame * 0.05) + 1) * 0.1 + 0.3
        color_intensity = int(40 + alpha * 20)
        cv2.line(frame, (0, i), (width, i), (color_intensity, color_intensity, color_intensity), 1)
    
    # System status indicator with pulsing effect
    if system_paused:
        pulse = int((math.sin(animation_frame * 0.2) + 1) * 50 + 100)
        status_color = (0, pulse, pulse)
        status_text = "SYSTEM PAUSED"
        cv2.putText(frame, status_text, (width//2 - 100, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    else:
        # Active system with animated title
        pulse = int((math.sin(animation_frame * 0.1) + 1) * 30 + 200)
        title_color = (pulse, 255, pulse)
        cv2.putText(frame, "Face Recognition System", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, title_color, 3)
        
        # Scanning indicator
        if detected_count == 0:
            scan_text = "Scanning..."
            scan_pulse = int((math.sin(animation_frame * 0.3) + 1) * 100 + 100)
            cv2.putText(frame, scan_text, (20, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (scan_pulse, scan_pulse, 255), 2)
    
    # Current time with enhanced styling
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, current_time, (width - 150, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['white'], 2)
    
    # Enhanced bottom info bar with animations
    info_y = height - 80
    for i in range(80):
        alpha = 0.6 + (math.sin(animation_frame * 0.03 + i * 0.1) + 1) * 0.1
        color_intensity = int(30 + alpha * 20)
        cv2.line(frame, (0, info_y + i), (width, info_y + i), (color_intensity, color_intensity, color_intensity), 1)
    
    # Statistics with current session info
    stats_line1 = f"Trained: {total_trained} | Currently Detected: {detected_count} | Currently Recognized: {recognized_count}"
    cv2.putText(frame, stats_line1, (20, info_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['white'], 2)
    
    # Session stats
    session_time = time.time() - session_stats['session_start']
    if session_stats['detected_names']:
        names_text = f"Session People: {', '.join(sorted(session_stats['detected_names']))}"
    else:
        names_text = "Session People: None detected yet"
    
    cv2.putText(frame, names_text, (20, info_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['light_blue'], 1)
    
    # Enhanced controls with colored keys
    controls_text = "Q:Quit | F:Fullscreen | P:Pause | R:Reset"
    text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(frame, controls_text, (width - text_size[0] - 20, info_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['yellow'], 2)
    
    return frame

def draw_animated_face_box(frame, x, y, w, h, name, confidence, is_recognized):
    """Draw animated face detection box with enhanced effects"""
    global animation_frame
    
    # Choose color and create pulsing effect
    if is_recognized:
        pulse = int((math.sin(animation_frame * 0.15) + 1) * 30 + 200)
        color = (0, pulse, 0)
        accent_color = COLORS['green']
    else:
        pulse = int((math.sin(animation_frame * 0.2) + 1) * 50 + 150)
        color = (0, 0, pulse)
        accent_color = COLORS['red']
    
    # Main rectangle with animated thickness
    thickness = int((math.sin(animation_frame * 0.1) + 1) * 2 + 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    
    # Animated corner accents
    corner_length = int(25 + (math.sin(animation_frame * 0.12) + 1) * 5)
    corner_thickness = 5
    
    # Glowing corner effects
    for offset in range(3):
        alpha = 1 - (offset * 0.3)
        glow_color = tuple(int(c * alpha) for c in accent_color)
        
        # Top-left corner
        cv2.line(frame, (x-offset, y-offset), (x + corner_length + offset, y-offset), glow_color, corner_thickness-offset)
        cv2.line(frame, (x-offset, y-offset), (x-offset, y + corner_length + offset), glow_color, corner_thickness-offset)
        
        # Top-right corner
        cv2.line(frame, (x + w + offset, y-offset), (x + w - corner_length - offset, y-offset), glow_color, corner_thickness-offset)
        cv2.line(frame, (x + w + offset, y-offset), (x + w + offset, y + corner_length + offset), glow_color, corner_thickness-offset)
        
        # Bottom-left corner
        cv2.line(frame, (x-offset, y + h + offset), (x + corner_length + offset, y + h + offset), glow_color, corner_thickness-offset)
        cv2.line(frame, (x-offset, y + h + offset), (x-offset, y + h - corner_length - offset), glow_color, corner_thickness-offset)
        
        # Bottom-right corner
        cv2.line(frame, (x + w + offset, y + h + offset), (x + w - corner_length - offset, y + h + offset), glow_color, corner_thickness-offset)
        cv2.line(frame, (x + w + offset, y + h + offset), (x + w + offset, y + h - corner_length - offset), glow_color, corner_thickness-offset)
    
    # Enhanced name label with glow effect
    label_text = f"{name.upper()}"
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    
    # Position label above the face box
    label_y = max(y - 50, 10)
    label_width = text_size[0] + 30
    
    # Animated background with gradient
    for i in range(40):
        alpha = 1 - (i / 40) * 0.5
        bg_color = tuple(int(c * alpha) for c in accent_color)
        cv2.rectangle(frame, (x-5, label_y + i), (x + label_width + 5, label_y + i + 1), bg_color, -1)
    
    # Add glowing text effect
    for offset in [(2,2), (1,1), (0,0)]:
        text_color = COLORS['white'] if offset == (0,0) else tuple(int(c * 0.5) for c in accent_color)
        cv2.putText(frame, label_text, (x + 10 + offset[0], label_y + 25 + offset[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    # Animated confidence meter
    if is_recognized:
        # Confidence bar
        bar_width = w
        bar_height = 8
        confidence_ratio = max(0, (100 - confidence) / 100)  # Invert confidence for visual appeal
        
        # Background bar
        cv2.rectangle(frame, (x, y + h + 10), (x + bar_width, y + h + 10 + bar_height), COLORS['dark_gray'], -1)
        
        # Animated confidence fill
        fill_width = int(bar_width * confidence_ratio)
        if confidence_ratio > 0.7:
            bar_color = COLORS['green']
        elif confidence_ratio > 0.4:
            bar_color = COLORS['yellow']
        else:
            bar_color = COLORS['orange']
            
        cv2.rectangle(frame, (x, y + h + 10), (x + fill_width, y + h + 10 + bar_height), bar_color, -1)
        
        # Confidence text
        conf_text = f"Confidence: {confidence:.1f}"
        cv2.putText(frame, conf_text, (x, y + h + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
    
    return frame
