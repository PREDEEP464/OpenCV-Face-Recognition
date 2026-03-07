import cv2
import time
import numpy as np

from Data_Loader import prepare_training_data
from LBPH_Recognizer import create_recognizer, train_recognizer, predict
import App_UI as ui

# Review 2 Enhancement: BMC for better recognition
try:
    from BMC_Processor import BMCProcessor
    BMC_AVAILABLE = True
    print("✅ BMC Enhancement Module Loaded")
except ImportError:
    BMC_AVAILABLE = False
    print("⚠️ BMC not available, using standard recognition")

def main():
    # Print fancy header
    ui.print_fancy_header()
    
    # Initialize BMC if available
    bmc_processor = None
    if BMC_AVAILABLE:
        print("="*80)
        print("🚀 REVIEW 2 ENHANCEMENT: BMC Integration")
        print("   ✓ Bilateral Median Convolution for robust recognition")
        print("   ✓ Better handling of lighting and occlusions")
        print("="*80)
        bmc_processor = BMCProcessor(kernel_size=5, sigma_space=75, sigma_color=75)

    print("📚 Preparing training data...")
    faces, labels, names = prepare_training_data()

    if faces is not None and labels is not None and names is not None:
        print("=" * 50)
        print("🧠 Training the face recognizer...")
        
        # Create and train recognizer (standard LBPH)
        face_recognizer = create_recognizer()
        train_recognizer(face_recognizer, faces, labels)
        
        print("✅ Training completed successfully!")
        print(f"👥 Trained on {len(names)} people: {', '.join(names)}")
        print("=" * 50)
        print("\n")
        
        # Initialize video capture with better settings
        print("📹 Initializing camera system...")
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Create fullscreen window
        ui.make_fullscreen()
        
        # Show camera initialization loader
        ui.animated_loader(video_capture, "Initializing Camera...", 3.0, ui.COLORS['blue'])
        print("✅ Camera initialization complete!")
        
        print("="*80)
        print("🚀 SYSTEM READY! Camera feed starting...")
        if BMC_AVAILABLE:
            print("📊 BMC preprocessing active (light mode for compatibility)")
        print("🎮 CONTROLS:")
        print("   Q - Quit System")
        print("   F - Toggle Fullscreen")
        print("   P - Pause/Resume Recognition")
        print("   R - Reset Statistics")
        if BMC_AVAILABLE:
            print("   T - Toggle BMC Statistics")
        print("="*80)
        
        # Initialize session stats and face cascade
        session_stats = {
            'detected_names': set(),
            'session_start': time.time(),
            'last_reset': time.time()
        }
        
        # Initialize BMC statistics tracking
        show_bmc_stats = False
        bmc_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_time_ms': 0.0,
            'recognitions_with_bmc': 0,
            'recognitions_without_bmc': 0
        }
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        system_paused = False
        
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                print("❌ Failed to capture frame from camera")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            detected_count = 0
            recognized_count = 0
            
            if not system_paused:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)  # Enhance contrast
                
                # Detect faces with optimized parameters
                faces_rect = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=8,
                    minSize=(80, 80),
                    maxSize=(400, 400),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                detected_count = len(faces_rect)
                
                # Process each detected face
                for (x, y, w, h) in faces_rect:
                    # Extract face region
                    face = gray[y:y+h, x:x+w]
                    
                    # Resize to match training data
                    face = cv2.resize(face, (200, 200))
                    
                    # Apply BMC enhancement if available (Review 2 enhancement - light mode)
                    if BMC_AVAILABLE and bmc_processor is not None:
                        bmc_start = time.time()
                        face = bmc_processor.fast_bmc(face, strength='light')
                        bmc_time = (time.time() - bmc_start) * 1000
                        bmc_stats['total_processed'] += 1
                        bmc_stats['total_time'] += bmc_time
                        bmc_stats['avg_time_ms'] = bmc_stats['total_time'] / bmc_stats['total_processed']
                    
                    # Perform recognition
                    label, confidence = predict(face_recognizer, face)
                    
                    # Check confidence threshold
                    if confidence < 70 and label < len(names):
                        name = names[label]
                        if BMC_AVAILABLE:
                            bmc_stats['recognitions_with_bmc'] += 1
                        is_recognized = True
                        recognized_count += 1
                        session_stats['detected_names'].add(name)
                    else:
                        name = "Unknown"
                        is_recognized = False
                    
                    # Adjust confidence for display
                    display_confidence = max(0, confidence - 42)
                    
                    # Draw animated face box
                    frame = ui.draw_animated_face_box(frame, x, y, w, h, name, display_confidence, is_recognized)
            
            # Add enhanced UI with all animations
            frame = ui.add_enhanced_ui(frame, detected_count, recognized_count, len(names), 
                                       session_stats, system_paused, show_bmc_stats, bmc_stats)
            
            # Display the frame
            cv2.imshow(ui.WINDOW_NAME, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Shutting down system...")
                # Show quit loader
                ui.animated_loader(video_capture, "Quitting System...", 3.0, ui.COLORS['red'])
                break
            elif key == ord('f'):
                # Toggle fullscreen
                prop = cv2.getWindowProperty(ui.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(ui.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("🪟  Switched to windowed mode")
                else:
                    cv2.setWindowProperty(ui.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("🖥️  Switched to fullscreen mode")
            elif key == ord('p'):
                # Toggle pause with loader only for resume
                if not system_paused:
                    print("⏸️  System PAUSED - Recognition stopped")
                    system_paused = True
                else:
                    print("▶️  Resuming system...")
                    ui.animated_loader(video_capture, "Resuming System...", 2.0, ui.COLORS['green'])
                    system_paused = False
                    print("▶️  System RESUMED - Recognition active")
            elif key == ord('r'):
                # Reset statistics with loader
                print("🔄 Resetting statistics...")
                ui.animated_loader(video_capture, "Resetting System...", 5.0, ui.COLORS['purple'])
                session_stats['detected_names'].clear()
                session_stats['last_reset'] = time.time()
                if BMC_AVAILABLE:
                    bmc_stats['total_processed'] = 0
                    bmc_stats['total_time'] = 0.0
                    bmc_stats['avg_time_ms'] = 0.0
                    bmc_stats['recognitions_with_bmc'] = 0
                print("🔄 Statistics RESET - Names list cleared")
            elif key == ord('t') and BMC_AVAILABLE:
                # Toggle BMC statistics display
                show_bmc_stats = not show_bmc_stats
                if show_bmc_stats:
                    print("📊 BMC Statistics: ON")
                else:
                    print("📊 BMC Statistics: OFF")
        
        # Cleanup and session summary
        video_capture.release()
        cv2.destroyAllWindows()
        
        ui.print_session_stats(session_stats)
    else:
        print("❌ Failed to prepare training data.")
        print("📝 Please check your Face_DB folder and ensure it contains valid image files.")
        print("💡 Supported formats: .jpg, .jpeg, .png, .bmp")


if __name__ == '__main__':
    main()
