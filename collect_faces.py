import cv2
import os
import subprocess
import sys

# Step 1: Ask for the person's name
name = input("Enter the name of the person: ").strip()

if not name:
    print("❌ Name cannot be empty.")
    exit()

# Step 2: Create the folder for the new person
save_path = os.path.join("dataset", name)
os.makedirs(save_path, exist_ok=True)

# Step 3: Open the webcam
cap = cv2.VideoCapture(0)
count = 0

print(f"[INFO] Capturing images for: {name}")
print("👉 Press 's' to save an image")
print("👉 Press 'q' to finish capturing")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        img_path = os.path.join(save_path, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[📸] Saved: {img_path}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Step 4: Automatically encode all faces
print("\n[INFO] Encoding all faces...")
subprocess.run([sys.executable, "encode_faces.py"])
print("[✅] Done! You can now run the attendance system.")
