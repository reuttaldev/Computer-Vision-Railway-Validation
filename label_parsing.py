import json
from pathlib import Path

def save_predictions_json(out_root, video_folder, image_path, frame, predictions):
    out_dir = out_root / video_folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{image_path.stem}.json"
    out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

def load_predictions(json_path):
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# match Railgoerl24 original annotations to the model's predictions
def match(gt_boxes, preds, iou_thr=0.5):
    matched_preds = set()
    tp = 0

    for gt in gt_boxes:
        best_iou = 0
        best_idx = -1

        for i, p in enumerate(preds):
            if i in matched_preds:
                continue
            score = iou(gt, p["bbox"])
            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_iou >= iou_thr:
            tp += 1
            matched_preds.add(best_idx)

    fn = len(gt_boxes) - tp
    fp = len(preds) - len(matched_preds)

    return tp, fp, fn
