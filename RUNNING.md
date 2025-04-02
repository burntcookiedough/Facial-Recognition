# How to Run the Facial Recognition Attendance System

This guide provides step-by-step instructions for running all the scripts in the system.

## Setup

Before using the system, run the setup script to ensure all dependencies and directories are properly configured:

```bash
python setup.py
```

## Using the System

### 1. Collect Face Images

To add a new person to the system:

```bash
# Basic usage
python -m scripts.collect

# With additional parameters
python -m scripts.collect --name "John Doe" --count 20
```

Parameters:
- `--name`: Name of the person (will prompt if not provided)
- `--output`: Custom directory to save images in
- `--count`: Number of images to collect

During collection:
- Press 's' to save an image
- Press 'q' to quit when done

### 2. Encode Faces

After collecting face images, encode them:

```bash
# Basic usage
python -m scripts.encode

# With custom paths
python -m scripts.encode --dataset custom/dataset/path --output custom_encodings.pickle
```

Parameters:
- `--dataset`: Path to the dataset directory
- `--output`: Path to save the encodings

### 3. Run the Attendance System

Start the face recognition attendance system:

```bash
# Basic usage
python -m scripts.attendance

# With custom settings
python -m scripts.attendance --encodings custom_encodings.pickle --tolerance 0.6
```

Parameters:
- `--encodings`: Path to the encodings file
- `--tolerance`: Face recognition tolerance (lower is stricter, range 0-1)

Press 'q' to exit the attendance system.

### 4. Face Detection Only

If you want to run face detection without recognition:

```bash
# Basic usage
python -m scripts.detect

# With custom settings
python -m scripts.detect --confidence 0.7
```

Parameters:
- `--confidence`: Detection confidence threshold (default 0.5)
- `--prototxt`: Path to the Caffe prototxt file
- `--model`: Path to the Caffe model file

Press 'q' to exit face detection.

## Accessing Help

All scripts support the `--help` flag to display available options:

```bash
python -m scripts.collect --help
python -m scripts.encode --help
python -m scripts.attendance --help
python -m scripts.detect --help
```

## Troubleshooting

If you encounter issues:

1. **Missing dependencies**: Run `python setup.py` again to install required packages
2. **Webcam not detected**: Check your camera connection and permissions
3. **Face not recognized**: Try creating more face images in different conditions
4. **Script errors**: Check the detailed error message for guidance

## Output Files

- Attendance logs are saved as CSV files named `attendance_YYYY-MM-DD.csv`
- Face encodings are saved in `encodings.pickle`
- Face images are stored in the `dataset/[name]` directories 