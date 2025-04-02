#!/usr/bin/env python3
"""
Collect face images for facial recognition training.
This script captures images from a webcam for a given person.
"""
import cv2
import os
import sys
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_utils import encode_face_images

def collect_face_images(name=None, output_dir=None, count_target=None):
    """
    Collect face images from webcam for a specified person.
    
    Args:
        name (str, optional): Name of the person to collect images for.
        output_dir (str, optional): Directory to save images in.
        count_target (int, optional): Number of images to collect (0 for unlimited).
    """
    # If name is not provided, ask for it
    if name is None or name.strip() == "":
        name = input("Enter the name of the person: ").strip()

    if not name:
        print("❌ Name cannot be empty.")
        return

    # Create the folder for the new person
    if output_dir is None:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    else:
        dataset_path = output_dir
        
    save_path = os.path.join(dataset_path, name)
    os.makedirs(save_path, exist_ok=True)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam. Please check your camera connection.")
        return

    count = 0
    print(f"[INFO] Capturing images for: {name}")
    if count_target:
        print(f"[INFO] Target: {count_target} images")
    print("👉 Press 's' to save an image")
    print("👉 Press 'q' to finish capturing")

    try:
        while True:
            # Check if we've reached the target count
            if count_target and count >= count_target:
                print(f"[INFO] Reached target of {count_target} images.")
                break
                
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame. Retrying...")
                time.sleep(0.5)
                continue

            # Display the frame with count
            if count_target:
                cv2.putText(frame, f"Images: {count}/{count_target}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Images: {count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.imshow("Capture Face", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                img_path = os.path.join(save_path, f"{name}_{count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"[📸] Saved: {img_path}")
                count += 1

            elif key == ord('q'):
                break

    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if count > 0:
        print(f"[INFO] Successfully captured {count} images.")
        
        # Ask if the user wants to encode all faces
        encode_now = input("Do you want to encode all faces now? (y/n): ").strip().lower()
        if encode_now == 'y':
            # Encode all faces
            print("\n[INFO] Encoding all faces...")
            encodings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encodings.pickle")
            total = encode_face_images(dataset_path, encodings_path)
            print(f"[✅] Done! Encoded {total} face images. You can now run the attendance system.")
        else:
            print("[INFO] You can encode faces later by running 'python scripts/encode.py'")
    else:
        print("⚠️ No images were captured.")
        
def main():
    """Parse arguments and run the face collection process."""
    parser = argparse.ArgumentParser(description="Collect face images for recognition")
    parser.add_argument("--name", type=str, help="Name of the person")
    parser.add_argument("--output", type=str, help="Directory to save images in")
    parser.add_argument("--count", type=int, help="Number of images to collect (0 for unlimited)")
    args = parser.parse_args()
    
    collect_face_images(args.name, args.output, args.count)

if __name__ == "__main__":
    main() 