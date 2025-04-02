#!/usr/bin/env python3
"""
Face recognition attendance system.
This script recognizes faces from a webcam and logs attendance.
"""
import cv2
import sys
import os
import argparse
from datetime import datetime
import csv
import time
import face_recognition

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_utils import load_encodings, get_attendance_file

def run_attendance_system(encodings_path=None, tolerance=0.5):
    """Run the face recognition attendance system."""
    if encodings_path is None:
        encodings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encodings.pickle")
    
    # Load encodings
    print(f"[INFO] Loading encodings from {encodings_path}...")
    data = load_encodings(encodings_path)
    
    if not data["encodings"]:
        print("❌ No face encodings found. Please run encode_faces.py first.")
        return
    
    print(f"[INFO] Loaded {len(data['encodings'])} face encodings.")
    
    # Set up attendance file
    attendance_file = get_attendance_file()
    logged_names = set()
    
    # Start webcam
    print("[INFO] Starting webcam...")
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("❌ Webcam not available. Please check your camera connection.")
        return
    
    print("[INFO] Attendance system started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("⚠️ Failed to grab frame. Retrying...")
                time.sleep(0.5)
                continue
            
            # Resize frame for faster processing (optional)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert to RGB (face_recognition uses RGB)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            
            if face_locations:
                # Generate encodings for the detected faces
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                # Process each detected face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=tolerance)
                    name = "Unknown"
                    
                    if True in matches:
                        # Find the best match
                        matched_idxs = [i for i, val in enumerate(matches) if val]
                        counts = {}
                        for i in matched_idxs:
                            matched_name = data["names"][i]
                            counts[matched_name] = counts.get(matched_name, 0) + 1
                        name = max(counts, key=counts.get)
                    
                    # Draw rectangle and name on the frame
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    
                    # Log attendance for recognized people (only once)
                    if name != "Unknown" and name not in logged_names:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(attendance_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([name, now])
                        logged_names.add(name)
                        print(f"[LOGGED] {name} at {now}")
            
            # Display the resulting frame
            cv2.imshow("Face Recognition Attendance", frame)
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
    
    print(f"[INFO] Attendance log saved to {attendance_file}")
    print(f"[INFO] Total attendance logged: {len(logged_names)} people")

def main():
    """Parse arguments and run the attendance system."""
    parser = argparse.ArgumentParser(description="Facial Recognition Attendance System")
    parser.add_argument("--encodings", type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encodings.pickle"),
                       help="Path to face encodings file")
    parser.add_argument("--tolerance", type=float, default=0.5,
                       help="Face recognition tolerance (lower is stricter, range 0-1)")
    args = parser.parse_args()
    
    # Run the attendance system
    run_attendance_system(args.encodings, args.tolerance)

if __name__ == "__main__":
    main() 