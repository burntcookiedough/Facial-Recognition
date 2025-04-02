#!/usr/bin/env python3
"""
Encode face images for facial recognition.
This script processes all images in the dataset directory and creates face encodings.
"""
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_utils import encode_face_images

def main():
    """Encode all face images in the dataset directory."""
    parser = argparse.ArgumentParser(description="Encode faces for recognition")
    parser.add_argument("--dataset", type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"),
                        help="Path to the dataset directory")
    parser.add_argument("--output", type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encodings.pickle"),
                        help="Path to save the encodings")
    args = parser.parse_args()
    
    # Verify dataset directory exists
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset directory not found: {args.dataset}")
        print("Creating the directory...")
        os.makedirs(args.dataset, exist_ok=True)
        print(f"✅ Created dataset directory: {args.dataset}")
        print("Please add face images before encoding.")
        return
    
    # Count number of subdirectories (people)
    people = [d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))]
    if not people:
        print("❌ No people found in the dataset directory.")
        print("Please run the collect_faces.py script first to gather face images.")
        return
    
    print(f"[INFO] Found {len(people)} people in the dataset.")
    
    # Encode all face images
    total = encode_face_images(args.dataset, args.output)
    
    print(f"[✅] Encoding complete! Processed {total} face images.")

if __name__ == "__main__":
    main() 