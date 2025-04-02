#!/usr/bin/env python3
"""
Common utilities for face recognition and detection.
This module provides helper functions for working with face recognition,
including loading and saving encodings, managing attendance logs,
setting up neural networks, and processing face images.
"""
import face_recognition
import cv2
import os
import pickle
from datetime import datetime
import csv

def load_encodings(encoding_file="encodings.pickle"):
    """
    Load face encodings from a pickle file.
    
    Args:
        encoding_file (str): Path to the pickle file containing encodings
        
    Returns:
        dict: Dictionary with 'encodings' and 'names' keys
    """
    try:
        with open(encoding_file, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"[ERROR] Encodings file {encoding_file} not found.")
        return {"encodings": [], "names": []}
    except Exception as e:
        print(f"[ERROR] Failed to load encodings: {e}")
        return {"encodings": [], "names": []}

def get_attendance_file():
    """
    Create and return the path to today's attendance file.
    
    Returns:
        str: Path to the attendance CSV file for today
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today_str}.csv"
    
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time"])
    
    return filename

def setup_dnn_network(prototxt_path, model_path):
    """
    Set up DNN network for face detection with proper backend.
    
    Args:
        prototxt_path (str): Path to the prototxt file
        model_path (str): Path to the caffemodel file
        
    Returns:
        cv2.dnn_Net: The configured neural network
    """
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        try:
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("[INFO] Using CUDA (FP16)")
        except Exception:
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[INFO] Using CUDA (default)")
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[INFO] CUDA not available. Using CPU fallback.")
    
    return net

def encode_face_images(dataset_path, encoding_file):
    """
    Encode all face images in the dataset directory.
    
    Args:
        dataset_path (str): Path to the directory containing face images
        encoding_file (str): Path where encodings should be saved
        
    Returns:
        int: Number of faces encoded
    """
    known_encodings = []
    known_names = []
    
    print("[INFO] Encoding faces...")
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue
            
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            print(f"[INFO] Processing image: {image_path}")
            
            try:
                image = cv2.imread(image_path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                boxes = face_recognition.face_locations(rgb, model="hog")
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)
            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {e}")
    
    data = {"encodings": known_encodings, "names": known_names}
    with open(encoding_file, "wb") as f:
        pickle.dump(data, f)
    
    print(f"[INFO] Encoded faces saved to {encoding_file}")
    return len(known_names)