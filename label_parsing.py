import json
from pathlib import Path

def save_predictions_json(out_root, video_folder, image_path, frame, predictions):
    out_dir = out_root / video_folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "image": image_path.name,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "predictions": predictions
    }

    out_path = out_dir / f"{image_path.stem}.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
def match_predictions(gt_boxes, preds, iou_thr=0.5):
    matched_gt = set()
    matched_pred = set()

    for gi, gt in enumerate(gt_boxes):
        best_iou = 0
        best_pi = -1

        for pi, p in enumerate(preds):
            if pi in matched_pred:
                continue

            i = iou(gt["bbox"], p["bbox"])
            if i > best_iou:
                best_iou = i
                best_pi = pi

        if best_iou >= iou_thr:
            matched_gt.add(gi)
            matched_pred.add(best_pi)

    tp = len(matched_gt)
    fn = len(gt_boxes) - tp
    fp = len(preds) - len(matched_pred)

    return tp, fp, fn
