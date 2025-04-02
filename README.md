# Facial Recognition Attendance System

A Python-based facial recognition system for tracking attendance automatically using computer vision.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.6%2B-blue)

## Overview

This system uses facial recognition to automatically record attendance. It captures faces through a webcam, 
matches them against a database of known faces, and logs attendance with timestamps.

The system is designed for:
- Educational institutions (schools, universities)
- Corporate environments
- Events and conferences
- Any setting requiring attendance tracking

## Features

- **Face Collection**: Capture and save face images for recognition training
- **Face Encoding**: Create facial encodings for efficient recognition
- **Attendance System**: Recognize faces and log attendance with timestamps
- **Face Detection**: Real-time face detection using OpenCV DNN
- **CSV Export**: Attendance logs are saved in CSV format for easy import into spreadsheets
- **Command-line Interface**: Simple scripts to operate the system

## Project Structure

```
├── dataset/               # Stores face images for each person
├── models/                # Pre-trained models for face detection
├── scripts/               # Easy-to-use launcher scripts
│   ├── collect.py         # Launch face collection
│   ├── encode.py          # Launch face encoding
│   ├── attendance.py      # Launch attendance system
│   ├── detect.py          # Launch face detection
│   └── download_models.py # Download required model files
├── src/                   # Source code
│   ├── collect_faces.py   # Face collection implementation
│   ├── encode_faces.py    # Face encoding implementation
│   ├── recognize_faces.py # Attendance system implementation
│   └── detect_faces_live.py # Face detection implementation
├── utils/                 # Utility modules
│   └── face_utils.py      # Common face recognition utilities
├── setup.py               # Setup script for easy installation
└── requirements.txt       # Package dependencies
```

## Requirements

- Python 3.6+
- Webcam or camera device
- Dependencies listed in `requirements.txt`

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- A webcam connected to your computer
- For Windows users: Microsoft Visual C++ Build Tools (for dlib installation)

### Quick Setup (Recommended)

For a streamlined setup process, simply run:

```bash
python setup.py
```

This script will:
1. Create necessary directories
2. Install all required dependencies 
3. Download the face detection model files
4. Verify that everything is ready to use

The setup script includes robust error handling and will inform you of any issues encountered during installation.

### Manual Setup

If you prefer to set up the system manually:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/facial-recognition-attendance.git
   cd facial-recognition-attendance
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained models for face detection:
   ```bash
   python scripts/download_models.py
   ```
   
   This script will automatically download the required model files. If it fails, you can:
   - Download the files manually from [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
   - Place `deploy.prototxt` and `res10_300x300_ssd_iter_140000_fp16.caffemodel` in the `models/` directory

## Quick Start

### 1. Collect Face Images

To add a new person to the system:

```bash
python scripts/collect.py
```

Follow the prompts to:
- Enter the person's name
- Capture multiple face images (press 's' to save an image)
- Press 'q' when done capturing

### 2. Encode Faces

After collecting face images, encode them:

```bash
python scripts/encode.py
```

This creates a database of facial features for recognition.

### 3. Run the Attendance System

Start the attendance recognition system:

```bash
python scripts/attendance.py
```

The system will:
- Detect faces in the webcam feed
- Recognize known individuals
- Log attendance with timestamps in a CSV file

### 4. View Attendance Records

Attendance records are saved as CSV files named `attendance_YYYY-MM-DD.csv` in the project root.

## Advanced Usage

### Face Detection Only

To run face detection without recognition:

```bash
python scripts/detect.py
```

### Command Line Arguments

All scripts support additional arguments. Here are some examples:

```bash
# Collect faces for a specific person
python scripts/collect.py --name "John Doe" --count 20

# Use custom paths for encoding
python scripts/encode.py --dataset custom/dataset/path --output custom_encodings.pickle

# Run attendance with custom settings
python scripts/attendance.py --encodings custom_encodings.pickle --tolerance 0.6

# Run face detection with custom model files
python scripts/detect.py --prototxt custom/deploy.prototxt --model custom/model.caffemodel
```

Use `--help` with any script to see all available options:

```bash
python scripts/collect.py --help
```

## Troubleshooting

### Common Issues

- **Webcam not detected**:
  - Ensure your webcam is properly connected and not in use by another application
  - Try using a different USB port
  - Check Device Manager (Windows) or System Information (macOS) to confirm the webcam is detected

- **Face not recognized**:
  - Try adding more face images of the person from different angles and lighting conditions
  - Adjust the tolerance parameter (e.g., `--tolerance 0.6`); higher values are more lenient
  - Ensure the person's face is well-lit and clearly visible

- **Performance issues**:
  - Use a machine with better hardware if possible
  - For slower machines, try using the HOG detection method instead of CNN
  - Adjust the frame processing rate in the code for smoother performance

### Installation Problems

- **dlib installation errors**:
  - Windows: Install Visual C++ Build Tools
  - macOS: Install XCode Command Line Tools
  - Linux: Install build-essential and cmake

  ```bash
  # Windows (with Chocolatey)
  choco install visualstudio2019buildtools

  # macOS
  xcode-select --install

  # Ubuntu/Debian
  sudo apt-get install build-essential cmake
  ```

- **OpenCV issues**:
  - If you encounter DLL load errors, try reinstalling OpenCV:
  ```bash
  pip uninstall opencv-python
  pip install opencv-python --no-cache-dir
  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- OpenCV and its contributors 
- dlib library and its machine learning algorithms 