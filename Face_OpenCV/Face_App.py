import cv2
import time
import numpy as np

from Data_Loader import prepare_training_data
from LBPH_Recognizer import create_recognizer, train_recognizer, predict
import App_UI as ui

def main():
    # Print fancy header exactly like original
    ui.print_fancy_header()

    print("📚 Preparing training data...")
    faces, labels, names = prepare_training_data()

    if faces is not None and labels is not None and names is not None:
        print("\n🧠 Training the face recognizer...")
        
        # Create and train recognizer
        face_recognizer = create_recognizer()
        train_recognizer(face_recognizer, faces, labels)
        
        print("✅ Training completed successfully!")
        print(f"👥 Trained on {len(names)} people: {', '.join(names)}")
        print("\n🎥 Initializing camera system...")
        
        # Initialize video capture with better settings
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Create fullscreen window
        ui.make_fullscreen()
        
        print("="*80)
        print("🚀 SYSTEM READY! Camera feed starting...")
        print("🎮 CONTROLS:")
        print("   Q - Quit System")
        print("   F - Toggle Fullscreen")
        print("   P - Pause/Resume Recognition")
        print("   R - Reset Statistics")
        print("="*80)
        
        # Show camera initialization loader
        print("📹 Initializing camera system...")
        ui.animated_loader(video_capture, "Initializing Camera...", 5.0, ui.COLORS['blue'])
        print("✅ Camera initialization complete!")
        
        # Initialize session stats and face cascade
        session_stats = {
            'detected_names': set(),
            'session_start': time.time(),
            'last_reset': time.time()
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
                
                # Detect faces with optimized parameters for stability
                faces_rect = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,     # Slower but more accurate
                    minNeighbors=8,      # Higher value = fewer false positives
                    minSize=(80, 80),    # Larger minimum size
                    maxSize=(400, 400),  # Maximum size to avoid huge detections
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                detected_count = len(faces_rect)
                
                # Process each detected face
                for (x, y, w, h) in faces_rect:
                    # Extract face region
                    face = gray[y:y+h, x:x+w]
                    
                    # Resize to match training data
                    face = cv2.resize(face, (200, 200))
                    
                    # Perform recognition
                    label, confidence = predict(face_recognizer, face)
                    
                    # Enhanced confidence threshold with multiple levels
                    if confidence < 70 and label < len(names):  # Optimized threshold
                        name = names[label]
                        is_recognized = True
                        recognized_count += 1
                        # Add recognized name to session stats
                        session_stats['detected_names'].add(name)
                    else:
                        name = "Unknown"
                        is_recognized = False
                    
                    # Adjust confidence for display purposes
                    display_confidence = max(0, confidence - 30) 
                    
                    # Draw animated face box with all the fancy effects
                    frame = ui.draw_animated_face_box(frame, x, y, w, h, name, display_confidence, is_recognized)
            
            # Add enhanced UI with all animations
            frame = ui.add_enhanced_ui(frame, detected_count, recognized_count, len(names), session_stats, system_paused)
            
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
                print("🔄 Statistics RESET - Names list cleared")
        
        # Cleanup and session summary
        video_capture.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*80)
        print("🏁 SESSION COMPLETED")
        ui.print_session_stats(session_stats)
        print("🙏 Thank you for using Advanced Face Recognition System!")
        print("="*80)
    else:
        print("❌ Failed to prepare training data.")
        print("📝 Please check your Face_DB folder and ensure it contains valid image files.")
        print("💡 Supported formats: .jpg, .jpeg, .png, .bmp")


if __name__ == '__main__':
    main()
