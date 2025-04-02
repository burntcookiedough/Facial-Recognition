#!/usr/bin/env python3
"""
Launcher script for attendance system.
This script is a convenient wrapper around src/recognize_faces.py,
allowing users to run the attendance system from the project root.
"""
import os
import sys
import subprocess
import argparse

def main():
    """
    Launch the face recognition attendance script with command-line arguments forwarding.
    
    Returns:
        int: Exit code from the target script
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    project_dir = os.path.dirname(script_dir)
    
    # Path to the target script
    target_script = os.path.join(project_dir, "src", "recognize_faces.py")
    
    # Check if the script exists
    if not os.path.exists(target_script):
        print(f"❌ Target script not found: {target_script}")
        return 1
    
    # Parse arguments to forward
    parser = argparse.ArgumentParser(description="Run facial recognition attendance system")
    parser.add_argument("--encodings", type=str, help="Path to face encodings file")
    parser.add_argument("--tolerance", type=float, help="Face recognition tolerance (lower is stricter, range 0-1)")
    args, unknown_args = parser.parse_known_args()
    
    # Build command with arguments
    cmd = [sys.executable, target_script]
    
    if args.encodings:
        cmd.extend(["--encodings", args.encodings])
        
    if args.tolerance:
        cmd.extend(["--tolerance", str(args.tolerance)])
    
    # Add any unknown args
    if unknown_args:
        cmd.extend(unknown_args)
    
    # Launch the script
    print("[INFO] Launching attendance system...")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 