#!/usr/bin/env python3
"""
Live face detection using OpenCV DNN.
This script detects faces in a live webcam feed using OpenCV's DNN module.
"""
import cv2
import os
import sys
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_utils import setup_dnn_network

def run_face_detection(prototxt=None, model=None, confidence_threshold=0.5):
    """Run live face detection using OpenCV DNN."""
    # Set default paths if not provided
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if prototxt is None:
        prototxt = os.path.join(base_dir, "models", "deploy.prototxt")
    if model is None:
        model = os.path.join(base_dir, "models", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    
    # Check if model files exist
    if not os.path.exists(prototxt):
        print(f"❌ Prototxt file not found: {prototxt}")
        return
    if not os.path.exists(model):
        print(f"❌ Model file not found: {model}")
        return
    
    # Load the DNN model
    print("[INFO] Loading face detection model...")
    net = setup_dnn_network(prototxt, model)
    
    # Open the webcam
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam. Please check your camera connection.")
        return
    
    print("[INFO] Face detection started. Press 'q' to quit.")
    
    # Try to get the first frame to test the model
    ret, frame = cap.read()
    if ret:
        try:
            # Test the model on the first frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 117.0, 123.0), False, False)
            net.setInput(blob)
            net.forward()
            print("[INFO] Model test successful.")
        except Exception as e:
            print(f"[ERROR] Model test failed: {e}")
            print("[INFO] Falling back to CPU.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    try:
        # Main detection loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame. Retrying...")
                time.sleep(0.5)
                continue
            
            # Create a blob from the frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 117.0, 123.0), False, False)
            net.setInput(blob)
            
            # Run detection
            detections = net.forward()
            
            # Process results
            h, w = frame.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    # Get face box coordinates
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence label
                    label = f"{confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display the frame with detections
            cv2.imshow("Face Detection (Press Q to quit)", frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("[INFO] Face detection completed.")

def main():
    """Parse arguments and run face detection."""
    parser = argparse.ArgumentParser(description="Live Face Detection")
    parser.add_argument("--prototxt", type=str, help="Path to the prototxt file")
    parser.add_argument("--model", type=str, help="Path to the Caffe model file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    
    run_face_detection(args.prototxt, args.model, args.confidence)

if __name__ == "__main__":
    main() 