import cv2

# Load the model
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Attempt to use CUDA backend with proper target handling
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    try:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        print("[INFO] Using CUDA (FP16)")
    except Exception as e:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("[INFO] Using CUDA (default)")
except Exception as e:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] CUDA not available. Using CPU fallback.")

# Open the webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

# --- Preliminary Test on First Frame ---
ret, frame = cap.read()
if ret:
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 117.0, 123.0), False, False)
    net.setInput(blob)
    try:
        test_detections = net.forward()
    except cv2.error as e:
        print("[ERROR] Forward pass failed with CUDA. Falling back to CPU.")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.setInput(blob)
        test_detections = net.forward()
        print("[INFO] Now using CPU fallback for inference.")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 117.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Detection (Press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
