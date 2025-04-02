#!/usr/bin/env python3
"""
Launcher script for face collection.
This script is a convenient wrapper around src/collect_faces.py,
allowing users to collect face images from the project root.
"""
import os
import sys
import subprocess
import argparse

def main():
    """
    Launch the face collection script with command-line arguments forwarding.
    
    Returns:
        int: Exit code from the target script
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    project_dir = os.path.dirname(script_dir)
    
    # Path to the target script
    target_script = os.path.join(project_dir, "src", "collect_faces.py")
    
    # Check if the script exists
    if not os.path.exists(target_script):
        print(f"❌ Target script not found: {target_script}")
        return 1
    
    # Parse arguments to forward
    parser = argparse.ArgumentParser(description="Collect face images for recognition")
    parser.add_argument("--name", type=str, help="Name of the person")
    parser.add_argument("--output", type=str, help="Path to save the images")
    parser.add_argument("--count", type=int, help="Number of images to collect")
    args, unknown_args = parser.parse_known_args()
    
    # Build command with arguments
    cmd = [sys.executable, target_script]
    
    if args.name:
        cmd.extend(["--name", args.name])
        
    if args.output:
        cmd.extend(["--output", args.output])
        
    if args.count:
        cmd.extend(["--count", str(args.count)])
    
    # Add any unknown args
    if unknown_args:
        cmd.extend(unknown_args)
    
    # Launch the script
    print("[INFO] Launching face collection tool...")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 