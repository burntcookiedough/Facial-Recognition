import face_recognition
import cv2
import pickle
import csv
from datetime import datetime
import os
import sys

# Get the absolute path to the folder where the .exe is located
base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Load encodings from correct path
encodings_path = os.path.join(base_dir, "encodings.pickle")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

# Set up attendance file
today_str = datetime.now().strftime("%Y-%m-%d")
filename = f"attendance_{today_str}.csv"

# Full path to the attendance file (in same folder as exe)
filename = os.path.join(base_dir, filename)

if not os.path.exists(filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

logged_names = set()

# Start webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ Webcam not available.")
    sys.exit()

print("[INFO] Attendance system started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model="hog")
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    names = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for i, val in enumerate(matches) if val]
            counts = {}
            for i in matched_idxs:
                matched_name = data["names"][i]
                counts[matched_name] = counts.get(matched_name, 0) + 1
            name = max(counts, key=counts.get)

        names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if name != "Unknown" and name not in logged_names:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, now])
            logged_names.add(name)
            print(f"[LOGGED] {name} at {now}")

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
