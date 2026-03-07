import cv2
import time
import numpy as np
import math
from datetime import datetime

# Modern Cyberpunk Color Palette (Review 2 Theme)
COLORS = {
    'neon_cyan': (180, 120, 0),      # Dark blue-cyan for primary elements (was bright cyan)
    'neon_pink': (255, 20, 147),     # Hot pink for recognized faces
    'neon_purple': (255, 0, 128),    # Purple for special effects
    'neon_green': (57, 255, 20),     # Lime green for success
    'neon_orange': (0, 140, 255),    # Orange for warnings
    'electric_blue': (180, 120, 0),  # Dark blue (was bright)
    'deep_purple': (128, 0, 128),    # Deep purple backgrounds
    'dark_bg': (15, 15, 25),         # Very dark background
    'gray_panel': (35, 35, 50),      # Panel background
    'white': (255, 255, 255),        # Pure white text
    'red_alert': (60, 60, 255),      # Red for unknown
    'gold': (0, 215, 255),           # Gold for highlights
}

WINDOW_NAME = '🎯 Face Recognition System'

# Global animation variables
animation_frame = 0
pulse_counter = 0

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

def add_enhanced_ui(frame, detected_count, recognized_count, total_trained, session_stats, system_paused, show_bmc_stats=False, bmc_stats=None):
    """Optimized UI with better performance"""
    global animation_frame, pulse_counter
    height, width = frame.shape[:2]
    animation_frame += 1
    pulse_counter += 0.05  # Reduced from 0.08 for less animation overhead
    
    # === LEFT SIDE PANEL (Simplified for performance) ===
    panel_w = 320
    panel_h = 200
    panel_x = 15
    panel_y = 15
    
    # Solid panel (no overlay for performance)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLORS['gray_panel'], -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLORS['neon_cyan'], 2)
    
    # Simple corner accents (reduced from 5px to 3px)
    accent_len = 25
    cv2.line(frame, (panel_x, panel_y), (panel_x + accent_len, panel_y), COLORS['neon_cyan'], 3)
    cv2.line(frame, (panel_x, panel_y), (panel_x, panel_y + accent_len), COLORS['neon_cyan'], 3)
    cv2.line(frame, (panel_x + panel_w, panel_y), (panel_x + panel_w - accent_len, panel_y), COLORS['neon_cyan'], 3)
    cv2.line(frame, (panel_x + panel_w, panel_y), (panel_x + panel_w, panel_y + accent_len), COLORS['neon_cyan'], 3)
    
    # Title with glow effect
    if system_paused:
        # Large centered paused message with styled background
        pause_w = 400
        pause_h = 100
        pause_x = (width - pause_w) // 2
        pause_y = (height - pause_h) // 2
        
        # Direct drawing - no overlay blending
        cv2.rectangle(frame, (pause_x, pause_y), (pause_x + pause_w, pause_y + pause_h), COLORS['gray_panel'], -1)
        
        # Animated red border
        pause_pulse = int((math.sin(pulse_counter * 2) + 1) * 80 + 120)
        pause_border = (60, 60, pause_pulse)
        cv2.rectangle(frame, (pause_x, pause_y), (pause_x + pause_w, pause_y + pause_h), pause_border, 4)
        
        # Paused text
        cv2.putText(frame, "SYSTEM PAUSED", (pause_x + 40, pause_y + 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (pause_pulse, pause_pulse, 255), 3)
        
        # Show in panel too
        title = "PAUSED"
        title_color = (pause_pulse, pause_pulse, 255)
    else:
        title = "FACE RECOGNITION"
        title_color = COLORS['neon_cyan']
    
    cv2.putText(frame, title, (panel_x + 15, panel_y + 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, title_color, 2)
    
    # Divider line
    cv2.line(frame, (panel_x + 10, panel_y + 45), (panel_x + panel_w - 10, panel_y + 45), COLORS['neon_cyan'], 1)
    
    # Statistics with icons
    stat_y = panel_y + 70
    line_h = 30
    
    # Trained count
    cv2.putText(frame, f"TRAINED: {total_trained}", (panel_x + 20, stat_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['neon_green'], 1)
    
    # Detected count with color coding
    detect_color = COLORS['neon_pink'] if detected_count > 0 else COLORS['white']
    cv2.putText(frame, f"DETECTED: {detected_count}", (panel_x + 20, stat_y + line_h), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, detect_color, 1)
    
    # Recognized count
    recog_color = COLORS['neon_green'] if recognized_count > 0 else COLORS['white']
    cv2.putText(frame, f"RECOGNIZED: {recognized_count}", (panel_x + 20, stat_y + line_h*2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, recog_color, 1)
    
    # Session people
    if session_stats['detected_names']:
        people_text = f"PEOPLE: {', '.join(list(session_stats['detected_names'])[:2])}"
        if len(session_stats['detected_names']) > 2:
            people_text += "..."
    else:
        people_text = "PEOPLE: Scanning..."
    cv2.putText(frame, people_text, (panel_x + 20, stat_y + line_h*3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['electric_blue'], 1)
    
    # === TOP RIGHT - Time and Status ===
    time_text = datetime.now().strftime("%H:%M:%S")
    time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    time_x = width - time_size[0] - 20
    time_y = 40
    # Shadow effect
    cv2.putText(frame, time_text, (time_x + 2, time_y + 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    # Grey-whitish color
    cv2.putText(frame, time_text, (time_x, time_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)
    
    # Scanning animation
    if not system_paused and detected_count == 0:
        scan_pulse = int((math.sin(pulse_counter * 3) + 1) * 100 + 100)
        cv2.putText(frame, "[SCANNING...]", (width - 200, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (scan_pulse, scan_pulse, 255), 1)
    
    # === BOTTOM CONTROLS BAR ===
    bar_h = 50
    bar_y = height - bar_h
    
    # Direct drawing - no overlay blending
    cv2.rectangle(frame, (0, bar_y), (width, height), COLORS['dark_bg'], -1)
    cv2.line(frame, (0, bar_y), (width, bar_y), COLORS['neon_cyan'], 2)
    
    # Controls
    controls = [
        ("Q", "QUIT", COLORS['red_alert']),
        ("F", "FULL", COLORS['electric_blue']),
        ("P", "PAUSE", COLORS['neon_orange']),
        ("R", "RESET", COLORS['neon_purple'])
    ]
    if bmc_stats is not None:
        controls.append(("B", "BMC", COLORS['neon_green']))
    
    x_offset = 30
    for key, label, color in controls:
        # Key button
        cv2.rectangle(frame, (x_offset, bar_y + 10), (x_offset + 30, bar_y + 35), color, 2)
        cv2.putText(frame, key, (x_offset + 8, bar_y + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Label
        cv2.putText(frame, label, (x_offset + 35, bar_y + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['white'], 1)
        x_offset += 120
    
    # === BMC STATISTICS PANEL (Bottom Right Corner) ===
    if show_bmc_stats and bmc_stats is not None:
        bmc_w = 320
        bmc_h = 235
        bmc_x = width - bmc_w - 15
        bmc_y = height - bmc_h - bar_h - 15  # Position above bottom bar
        
        # Futuristic panel with neon green border - direct drawing
        cv2.rectangle(frame, (bmc_x, bmc_y), (bmc_x + bmc_w, bmc_y + bmc_h), COLORS['gray_panel'], -1)
        
        # Animated green/cyan border
        bmc_pulse = int((math.sin(pulse_counter * 1.5) + 1) * 60 + 140)
        bmc_border = (bmc_pulse, 255, bmc_pulse)
        cv2.rectangle(frame, (bmc_x, bmc_y), (bmc_x + bmc_w, bmc_y + bmc_h), bmc_border, 3)
        
        # Diagonal accent lines
        for i in range(0, bmc_w, 40):
            line_alpha = 100 + int((math.sin(pulse_counter + i * 0.1) + 1) * 30)
            cv2.line(frame, (bmc_x + i, bmc_y), (bmc_x + i + 20, bmc_y + 20), (0, line_alpha, line_alpha), 1)
        
        # Title
        cv2.putText(frame, "BMC ANALYTICS", (bmc_x + 15, bmc_y + 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLORS['neon_green'], 2)
        cv2.line(frame, (bmc_x + 10, bmc_y + 40), (bmc_x + bmc_w - 10, bmc_y + 40), COLORS['neon_green'], 1)
        
        # Metrics
        metric_y = bmc_y + 62
        metric_spacing = 26
        
        # Frames processed
        frames_count = bmc_stats['total_processed']
        cv2.putText(frame, f"Frames Processed: {frames_count}", (bmc_x + 20, metric_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLORS['white'], 1)
        
        # Processing time with color coding
        avg_time = bmc_stats['avg_time_ms']
        if avg_time < 10:
            time_color = COLORS['neon_green']
            status_text = "GREEN"
        elif avg_time < 20:
            time_color = COLORS['neon_orange']
            status_text = "YELLOW"
        else:
            time_color = COLORS['red_alert']
            status_text = "RED"
        
        cv2.putText(frame, f"Avg Processing: {avg_time:.2f} ms", (bmc_x + 20, metric_y + metric_spacing), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, time_color, 1)
        cv2.putText(frame, f"[{status_text}]", (bmc_x + 230, metric_y + metric_spacing), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, time_color, 1)
        
        # BMC Recognition count
        bmc_recs = bmc_stats['recognitions_with_bmc']
        cv2.putText(frame, f"BMC Recognitions: {bmc_recs}", (bmc_x + 20, metric_y + metric_spacing*2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLORS['neon_green'], 1)
        
        # Status with pulsing indicator
        status_pulse = int((math.sin(pulse_counter * 2) + 1) * 50 + 150)
        cv2.circle(frame, (bmc_x + 20, metric_y + metric_spacing*3 - 3), 5, (0, status_pulse, 0), -1)
        cv2.putText(frame, "Status: ACTIVE (Light Mode)", (bmc_x + 32, metric_y + metric_spacing*3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLORS['neon_green'], 1)
        
        # Est. Throughput
        if avg_time > 0:
            throughput = 1000.0 / avg_time
            cv2.putText(frame, f"Est. Throughput: ~{throughput:.0f} FPS", (bmc_x + 20, metric_y + metric_spacing*4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLORS['electric_blue'], 1)
        
        # Enhancement percentage
        enhancement_pct = 0
        if frames_count > 0:
            enhancement_pct = (bmc_recs / frames_count) * 100
        
        enh_color = COLORS['neon_green'] if enhancement_pct > 50 else COLORS['neon_orange'] if enhancement_pct > 25 else COLORS['white']
        cv2.putText(frame, f"Output Enhanced: {enhancement_pct:.1f}%", (bmc_x + 20, metric_y + metric_spacing*5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, enh_color, 1)
        
        # System FPS (smaller, at bottom)
        actual_fps = bmc_stats.get('actual_fps', 0)
        fps_color = COLORS['neon_green'] if actual_fps > 20 else COLORS['neon_orange'] if actual_fps > 15 else COLORS['red_alert']
        cv2.putText(frame, f"System FPS: {actual_fps:.1f}", (bmc_x + 20, metric_y + metric_spacing*6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, fps_color, 1)
    
    return frame

def draw_animated_face_box(frame, x, y, w, h, name, confidence, is_recognized):
    """Simple clean face box with professional appearance"""
    global pulse_counter
    
    # Color selection
    if is_recognized:
        box_color = COLORS['neon_green']
        name_bg_color = (0, 180, 0)  # Dark green
    else:
        box_color = COLORS['red_alert']
        name_bg_color = (0, 0, 180)  # Dark red
    
    # Simple bounding box - clean outline only
    thickness = 2
    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, thickness)
    
    # Name label (simple rectangle above face)
    name_text = name.upper()
    text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    label_y = max(y - 35, 10)
    label_w = text_size[0] + 20
    label_h = 30
    label_x = x
    
    # Simple solid background
    cv2.rectangle(frame, (label_x, label_y), (label_x + label_w, label_y + label_h), name_bg_color, -1)
    cv2.rectangle(frame, (label_x, label_y), (label_x + label_w, label_y + label_h), box_color, 2)
    
    # Name text
    cv2.putText(frame, name_text, (label_x + 10, label_y + 21), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 2)
    
    # Confidence bar (clean horizontal bar)
    if is_recognized:
        bar_y = y + h + 10
        bar_w = w
        bar_h = 10
        
        # Background bar
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), box_color, 1)
        
        # Fill based on confidence (inverted - lower is better)
        confidence_ratio = max(0, (100 - confidence) / 100)
        fill_w = int(bar_w * confidence_ratio)
        
        # Gradient color based on quality
        if confidence_ratio > 0.7:
            fill_color = (0, 200, 0)  # Green
        elif confidence_ratio > 0.5:
            fill_color = (0, 200, 200)  # Yellow-green
        elif confidence_ratio > 0.3:
            fill_color = (0, 150, 255)  # Orange
        else:
            fill_color = (0, 0, 200)  # Red
        
        # Draw filled portion
        if fill_w > 0:
            cv2.rectangle(frame, (x, bar_y), (x + fill_w, bar_y + bar_h), fill_color, -1)
        
        # Confidence score text
        conf_text = f"Confidence: {confidence:.1f}"
        cv2.putText(frame, conf_text, (x, bar_y + bar_h + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
    
    return frame
