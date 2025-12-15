from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, conf=0.3, imgsz=960):
        self.model = YOLO("yolov8s.pt")
        self.conf = conf #confidence threshold
        self.imgsz = imgsz #image size

    def detect(self, frame):
        results = self.model(frame,verbose=False,conf=self.conf,imgsz=self.imgsz)

        predictions = []

        for frame_result in results:
            for box in frame_result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                predictions.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

        return predictions
