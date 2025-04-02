#!/usr/bin/env python3
"""
Setup script for Facial Recognition Attendance System.
This script:
1. Creates necessary directories
2. Installs dependencies
3. Downloads required model files
"""
import os
import sys
import subprocess
import importlib.util
import platform

def check_python_version():
    """
    Check if Python version is 3.6 or higher.
    
    Returns:
        bool: True if Python version is compatible
    """
    if sys.version_info < (3, 6):
        print("[❌] Python 3.6 or higher is required.")
        sys.exit(1)
    print(f"[✅] Python version: {sys.version.split()[0]}")
    print(f"[INFO] Running on: {platform.system()} {platform.release()}")
    return True

def create_directories():
    """
    Create necessary directories if they don't exist.
    
    Returns:
        bool: True if all directories were created or already exist
    """
    directories = ['dataset', 'models']
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"[✅] Created directory: {directory}")
            except Exception as e:
                print(f"[❌] Failed to create directory {directory}: {e}")
                return False
        else:
            print(f"[INFO] Directory already exists: {directory}")
    
    # Create .gitkeep files
    for directory in directories:
        gitkeep_file = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_file):
            try:
                with open(gitkeep_file, 'w') as f:
                    f.write('# This directory is managed by the setup.py script\n')
                print(f"[✅] Created {gitkeep_file}")
            except Exception as e:
                print(f"[⚠️] Failed to create {gitkeep_file}: {e}")
                # Non-critical error, continue
    
    return True

def install_dependencies():
    """
    Install dependencies from requirements.txt.
    
    Returns:
        bool: True if dependencies were installed successfully
    """
    requirements_file = 'requirements.txt'
    
    if not os.path.exists(requirements_file):
        print(f"[❌] {requirements_file} not found.")
        return False
    
    print(f"[INFO] Installing dependencies from {requirements_file}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        print("[INFO] Pip upgraded.")
        
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("[✅] Dependencies installed successfully.")
        
        # Check if critical packages are available
        critical_packages = ['opencv-python', 'face_recognition', 'dlib', 'numpy']
        missing_packages = []
        
        for package in critical_packages:
            if importlib.util.find_spec(package.replace('-', '_')) is None:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"[⚠️] Some critical packages may not be installed correctly: {', '.join(missing_packages)}")
            print("[INFO] You may need to install them manually.")
            return False
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"[❌] Failed to install dependencies: {e}")
        return False

def download_models():
    """
    Download required model files.
    
    Returns:
        bool: True if models were downloaded successfully
    """
    # Check if download_models.py script exists
    download_script = os.path.join('scripts', 'download_models.py')
    
    if not os.path.exists(download_script):
        print(f"[❌] {download_script} not found.")
        return False
    
    print("[INFO] Downloading model files...")
    try:
        subprocess.check_call([sys.executable, download_script])
        
        # Verify models were downloaded
        model_files = [
            os.path.join('models', 'deploy.prototxt'),
            os.path.join('models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        ]
        
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"[⚠️] The following model files are still missing: {', '.join(missing_files)}")
            print("[INFO] You may need to download them manually.")
            return False
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"[❌] Failed to download model files: {e}")
        return False

def main():
    """
    Run the setup process.
    
    Returns:
        int: 0 for success, 1 for failure
    """
    print("\n======= Facial Recognition Attendance System Setup =======\n")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Downloading models", download_models)
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n[STEP] {step_name}...")
        if not step_func():
            print(f"[⚠️] Step '{step_name}' failed or had warnings.")
            if step_name in ["Installing dependencies", "Downloading models"]:
                # These steps can fail but we can still continue
                success = False
            
    
    print("\n======= Setup " + ("Complete with Warnings" if not success else "Complete") + " =======")
    print("\nTo start using the system:")
    print("1. First collect face images: python scripts/collect.py")
    print("2. Then encode the faces: python scripts/encode.py")
    print("3. Run the attendance system: python scripts/attendance.py")
    print("\nFor face detection only: python scripts/detect.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 