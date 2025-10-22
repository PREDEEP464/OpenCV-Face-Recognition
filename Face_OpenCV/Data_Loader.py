import os
import cv2
import numpy as np

def prepare_training_data():
    """Prepare training data exactly like the original Face_OpenCV.py"""
    faces = []
    labels = []
    names = []
    
    # Face database folder path - handle both running from Face_OpenCV folder and parent folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    face_db_path = os.path.join(parent_dir, 'Face_DB')
    
    # Check if Face_DB folder exists
    if not os.path.exists(face_db_path):
        print(f"Face_DB folder not found at: {face_db_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script location: {current_dir}")
        return None, None, None
    
    # Get all image files from Face_DB folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(face_db_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("No image files found in Face_DB folder!")
        return None, None, None
    
    print(f"Found {len(image_files)} images in Face_DB folder:")
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Process each image file
    label_id = 0
    for image_file in image_files:
        img_path = os.path.join(face_db_path, image_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not load image: {image_file}")
            continue
        
        # Get person name from filename (without extension)
        person_name = os.path.splitext(image_file)[0]
        print(f"Processing {person_name}...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces_rect) == 0:
            print(f"No face found in image: {image_file}")
            continue
        
        # Use the first detected face (largest face if multiple)
        if len(faces_rect) > 1:
            # Sort faces by area (largest first)
            faces_rect = sorted(faces_rect, key=lambda x: x[2] * x[3], reverse=True)
        
        (x, y, w, h) = faces_rect[0]
        face = gray[y:y+h, x:x+w]
        
        # Resize to standard size for better recognition
        face = cv2.resize(face, (200, 200))
        
        faces.append(face)
        labels.append(label_id)
        names.append(person_name)
        
        print(f"Successfully processed {person_name} (Label: {label_id})")
        label_id += 1
    
    if not faces:
        print("No faces were successfully processed!")
        return None, None, None
    
    print(f"Training data prepared successfully for {len(faces)} faces!")
    return faces, labels, names
