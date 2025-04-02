import face_recognition
import os
import cv2
import pickle

dataset_path = "dataset"
encoding_file = "encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] Encoding faces...")

# Loop through each person's folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    # Loop through each image in the person's folder
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        print(f"[INFO] Processing image: {image_path}")

        # Load the image and convert it to RGB
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        boxes = face_recognition.face_locations(rgb, model="hog")  # "cnn" is more accurate but slower
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save encodings to a file
data = {"encodings": known_encodings, "names": known_names}
with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encoded faces saved to {encoding_file}")
