import cv2
import pyttsx3
import threading
import numpy as np

# Initialize TTS Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)

def speak_async(message):
    def speak():
        tts_engine.say(message)
        tts_engine.runAndWait()
    threading.Thread(target=speak, daemon=True).start()

# Load MobileNetSSD model
prototxt_path = "C:/mobile_-net_ssd project/MobileNetSSD_deploy.prototxt"
model_path = "C:/mobile_-net_ssd project/MobileNetSSD_deploy.caffemodel"


net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Classes MobileNetSSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle","bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("Press 'q' to quit.")

frame_counter = 0
last_spoken_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Prepare input
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                       frame.shape[1], frame.shape[0]])
            (x_min, y_min, x_max, y_max) = box.astype("int")

            # Draw box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Direction
            box_center_x = (x_min + x_max) // 2
            if box_center_x < frame.shape[1] // 3:
                direction = "left"
            elif box_center_x > 2 * frame.shape[1] // 3:
                direction = "right"
            else:
                direction = "straight ahead"

            # Speak once every 3 seconds
            current_time = cv2.getTickCount()
            time_elapsed = (current_time - last_spoken_time) / cv2.getTickFrequency()
            if time_elapsed > 3:
                message = f"Detected {label} {direction}."
                speak_async(message)
                last_spoken_time = current_time

    cv2.imshow("MobileNetSSD Fast Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
