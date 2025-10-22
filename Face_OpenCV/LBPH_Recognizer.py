import cv2
import numpy as np

# Face recognizer helper

def create_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    return recognizer

def train_recognizer(recognizer, faces, labels):
    recognizer.train(faces, np.array(labels))

def predict(recognizer, face):
    return recognizer.predict(face)
