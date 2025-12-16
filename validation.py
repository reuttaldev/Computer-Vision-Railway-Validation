from label_parsing import load_prediction,load_label
import matplotlib.pyplot as plt
from pathlib import Path

class Validator:
    def __init__(self, root_labels_dir, root_pred_dir, iou_thr=0.5, conf_thr = 0.5):
        # decides whether a prediction can be considered a TP candidate
        self.iou_thr = iou_thr
        # decides whether a prediction is trusted / counted
        self.conf_thr = conf_thr 
        self.root_labels_dir = root_labels_dir
        self.root_pred_dir = root_pred_dir

    # match Railgoerl24 original annotations to the model's predictions 
    def match(self, label, pred):
        # Tte order of the detected objects in the prediction and label is not necessarily the same 
        matched_preds = set()
            
        # for each person detected in the ground truth
        for gt in label:  
            max_iou = 0
            max_iou_index = -1

            for i, p in enumerate(pred):  
                # ensure one to one matching
                if i in matched_preds:
                    continue

                iou_val = self.iou(gt, p["bbox"])
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_iou_index = i

            # decide if the best score is good enough to be considered a tp
            # and if we are sure enough of the prediction
            if max_iou >= self.iou_thr and max_iou_index != -1 and pred[max_iou_index]["confidence"] >= self.conf_thr:
                self.tp += 1
                matched_preds.add(max_iou_index)
                self.ious.append(max_iou)
            else:  # meaning the model did not detect a person even when the label showed it there
                self.fn += 1

        # prediction that was not matched to a label is fp
        for i, p in enumerate(pred):   
            if i not in matched_preds and p["confidence"] >= self.conf_thr:
                self.fp += 1
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.ious = []

    def validate(self):
        self.reset()
        for pred_dir in self.root_pred_dir.iterdir():
            label_dir = self.root_labels_dir / f"{pred_dir.name}_auto_annots"
            if not label_dir.exists():
                print(f"No labels for {pred_dir.name}")
                continue

            for xml_path in label_dir.glob("*.xml"):
                label = load_label(xml_path)
                frame_name = xml_path.stem
                json_path = pred_dir / f"{frame_name}.json"
                pred = load_prediction(json_path)
                self.match(label, pred)

    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-9)

    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-9)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-9)
    
    def iou(self,boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

        return inter / (areaA + areaB - inter + 1e-6)
    
    def mean_iou(self):
        return sum(self.ious) / len(self.ious) if self.ious else 0.0

    def false_positive_rate(self):
        return self.fp / (self.fp + self.tp + 1e-9)

    def false_negative_rate(self):
        return self.fn / (self.tp + self.fn + 1e-9)
    
    def print(self):
        print("Precision:", self.precision())
        print("Recall:", self.recall())
        print("F1:", self.f1())
        print("Mean IoU:", self.mean_iou())
        print("FP rate:", self.false_positive_rate())
        print("FN rate:", self.false_negative_rate())

    def plot(self, thresholds, out_path="images/threshold.png"):
        xs, ps, rs, f1s = [], [], [], []

        for t in thresholds:
            self.conf_thr = t
            self.validate()
            self.print()
            xs.append(float(t))
            rs.append(float(self.recall()))
            ps.append(float(self.precision()))
            f1s.append(float(self.f1()))

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(xs, ps, label="Precision")
        plt.plot(xs, rs, label="Recall")
        plt.plot(xs, f1s, label="F1")
        plt.xlabel("Confidence threshold")
        plt.ylabel("Score")
        plt.title("Precision / Recall / F1 vs Threshold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.show()