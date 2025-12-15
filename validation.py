import cv2
from detector import ObjectDetector

detector = ObjectDetector("yolov8n.pt")

cap = cv2.VideoCapture("datasets/night/person_on_track.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    for d in detections:
        if d["label"] in ["person", "car"] and d["confidence"] > 0.6:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, d["label"], (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Validation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
