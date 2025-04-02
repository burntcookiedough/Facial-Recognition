#!/usr/bin/env python3
"""
Launcher script for face detection.
This script is a convenient wrapper around src/detect_faces_live.py,
allowing users to run face detection from the project root.
"""
import os
import sys
import subprocess
import argparse

def main():
    """
    Launch the face detection script with command-line arguments forwarding.
    
    Returns:
        int: Exit code from the target script
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    project_dir = os.path.dirname(script_dir)
    
    # Path to the target script
    target_script = os.path.join(project_dir, "src", "detect_faces_live.py")
    
    # Check if the script exists
    if not os.path.exists(target_script):
        print(f"❌ Target script not found: {target_script}")
        return 1
    
    # Parse arguments to forward
    parser = argparse.ArgumentParser(description="Run live face detection")
    parser.add_argument("--confidence", type=float, help="Confidence threshold")
    parser.add_argument("--prototxt", type=str, help="Path to Caffe 'deploy' prototxt file")
    parser.add_argument("--model", type=str, help="Path to Caffe pre-trained model")
    args, unknown_args = parser.parse_known_args()
    
    # Build command with arguments
    cmd = [sys.executable, target_script]
    
    if args.confidence:
        cmd.extend(["--confidence", str(args.confidence)])
        
    if args.prototxt:
        cmd.extend(["--prototxt", args.prototxt])
        
    if args.model:
        cmd.extend(["--model", args.model])
    
    # Add any unknown args
    if unknown_args:
        cmd.extend(unknown_args)
    
    # Launch the script
    print("[INFO] Launching face detection tool...")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 