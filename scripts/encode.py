#!/usr/bin/env python3
"""
Launcher script for face encoding.
This script is a convenient wrapper around src/encode_faces.py,
allowing users to encode faces from the project root.
"""
import os
import sys
import subprocess
import argparse

def main():
    """
    Launch the face encoding script with command-line arguments forwarding.
    
    Returns:
        int: Exit code from the target script
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    project_dir = os.path.dirname(script_dir)
    
    # Path to the target script
    target_script = os.path.join(project_dir, "src", "encode_faces.py")
    
    # Check if the script exists
    if not os.path.exists(target_script):
        print(f"❌ Target script not found: {target_script}")
        return 1
    
    # Parse arguments to forward
    parser = argparse.ArgumentParser(description="Encode faces for recognition")
    parser.add_argument("--dataset", type=str, help="Path to the dataset directory")
    parser.add_argument("--output", type=str, help="Path to save the encodings")
    args, unknown_args = parser.parse_known_args()
    
    # Build command with arguments
    cmd = [sys.executable, target_script]
    
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
        
    if args.output:
        cmd.extend(["--output", args.output])
    
    # Add any unknown args
    if unknown_args:
        cmd.extend(unknown_args)
    
    # Launch the script
    print("[INFO] Launching face encoding tool...")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 