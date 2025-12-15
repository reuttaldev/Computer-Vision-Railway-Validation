
from object_detection import PersonDetector
import cv2
from pathlib import Path
from label_parsing import save_predictions_json

DATA_DIR = Path(r"railgoerl24_dataset/videos")
PRED_DIR = Path(r"predictions")

# processe a folder of frames (video)
def run_sequence(detector,sequence_dir,show = True):

    image_paths = sorted(sequence_dir.glob("*.jpg"))

    if not image_paths:
        print("No images found in:", sequence_dir)
        return

    print(f"Running on sequence: {sequence_dir.name}")
    print(f"Total frames: {len(image_paths)}")

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        predictions = detector.detect(frame)
        if predictions:
            save_predictions_json(PRED_DIR,sequence_dir,img_path,frame,predictions) 

        # Filter persons
        persons = [
            d for d in predictions
            if d["label"] == "person" and d["confidence"] >= 0.3
        ]

        if show:
            # Draw detections
            for d in persons:
                x1, y1, x2, y2 = d["bbox"]
                conf = d["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {conf:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("YOLOv8s – pretrained (ESC to quit)", frame)

            # 30 ms delay ≈ video playback
            if cv2.waitKey(30) & 0xFF == 27:
                break




if __name__ == "__main__":
    detector = PersonDetector()
    seq_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()] )
    for seq_idx, seq_dir in enumerate(seq_dirs, start=1):
        r = run_sequence(detector,seq_dir,False)
    cv2.destroyAllWindows()
