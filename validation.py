from label_parsing import load_prediction,load_label

class Validator:
    def __init__(self, iou_thr=0.5):
        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.iou_thr = iou_thr
        self.ious = [] 
        self.confidence_matches = [] 

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
            if max_iou >= self.iou_thr and max_iou_index != -1:
                self.tp += 1
                matched_preds.add(max_iou_index)
                self.ious.append(max_iou)
                self.confidence_matches.append(
                    (pred[max_iou_index]["confidence"], True)
                )
            else:  # meaning the model did not detect a person even when the label showed it there
                self.fn += 1

        # prediction that was not matched to a label is fp
        for i, p in enumerate(pred):   
            if i not in matched_preds:
                self.fp += 1
                self.confidence_matches.append(
                    (p["confidence"], False)
                )

    def validate(self, root_labels_dir, root_pred_dir):
        for pred_dir in root_pred_dir.iterdir():
            label_dir = root_labels_dir / f"{pred_dir.name}_auto_annots"
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

    def map_50(self):
        if not self.confidence_matches:
            return 0.0

        data = sorted(self.confidence_matches, key=lambda x: -x[0])

        tp_cum = 0
        fp_cum = 0
        ap = 0.0
        prev_recall = 0.0
        total_gt = self.tp + self.fn

        for _, is_tp in data:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1

            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            recall = tp_cum / (total_gt + 1e-9)

            ap += precision * (recall - prev_recall)
            prev_recall = recall

        return ap
    
    def print(self):
        print("Precision:", self.precision())
        print("Recall:", self.recall())
        print("F1:", self.f1())
        print("mAP@0.5:", self.map_50())
        print("Mean IoU:", self.mean_iou())
        print("FP rate:", self.false_positive_rate())
        print("FN rate:", self.false_negative_rate())

    

