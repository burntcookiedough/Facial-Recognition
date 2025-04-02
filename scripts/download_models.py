#!/usr/bin/env python3
"""
Download script for face detection models.
This script downloads the needed model files for OpenCV DNN face detection.
"""
import os
import sys
import urllib.request
import ssl

def main():
    """Download model files if they don't exist."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    project_dir = os.path.dirname(script_dir)
    
    # Models directory
    models_dir = os.path.join(project_dir, "models")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        print(f"[INFO] Creating models directory: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
    
    # Files to download with primary and backup URLs
    files = {
        "deploy.prototxt": [
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt"
        ],
        "res10_300x300_ssd_iter_140000_fp16.caffemodel": [
            "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        ]
    }
    
    # Download files
    for filename, urls in files.items():
        file_path = os.path.join(models_dir, filename)
        
        if os.path.exists(file_path):
            print(f"[INFO] {filename} already exists. Skipping download.")
            continue
        
        print(f"[INFO] Downloading {filename}...")
        
        # Try primary URL first, then backup
        downloaded = False
        for i, url in enumerate(urls):
            try:
                print(f"[INFO] Attempting download from source {i+1}...")
                # Create SSL context that doesn't verify certificates
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(url, context=context) as response, open(file_path, 'wb') as out_file:
                    file_size = int(response.headers.get('Content-Length', 0))
                    if file_size > 0:
                        print(f"[INFO] File size: {file_size/1024/1024:.2f} MB")
                    
                    # Download with a simple progress indicator
                    downloaded_size = 0
                    block_size = 8192
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded_size += len(buffer)
                        out_file.write(buffer)
                        if file_size > 0:
                            percent = downloaded_size * 100 / file_size
                            print(f"\r[INFO] Downloaded: {downloaded_size/1024/1024:.2f} MB ({percent:.1f}%)", end='')
                
                print(f"\n[✅] Successfully downloaded {filename}")
                downloaded = True
                break
            except Exception as e:
                print(f"[⚠️] Failed to download from source {i+1}: {e}")
                continue
        
        if not downloaded:
            print(f"[❌] Failed to download {filename} from all sources.")
            print(f"     Please download manually from one of these URLs:")
            for i, url in enumerate(urls):
                print(f"     Source {i+1}: {url}")
            print(f"     And place it in: {models_dir}")
    
    # Check if all files were downloaded successfully
    all_files_exist = all(os.path.exists(os.path.join(models_dir, filename)) for filename in files)
    
    if all_files_exist:
        print("\n[✅] All model files are available. The system is ready to use!")
    else:
        print("\n[⚠️] Some model files are missing. Please download them manually.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 