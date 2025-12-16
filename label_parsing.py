import json
import xml.etree.ElementTree as ET

def load_label(path):
        tree = ET.parse(path)
        root = tree.getroot()

        boxes = []
        for tag in ["object", "unsure_objects"]:
            for obj in root.findall(tag):
                bb = obj.find("bndbox")
                boxes.append([
                    int(bb.find("xmin").text),
                    int(bb.find("ymin").text),
                    int(bb.find("xmax").text),
                    int(bb.find("ymax").text),
                ])
        return boxes

def save_predictions_json(path, video_folder, image_path, predictions):
    out_dir = path / video_folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{image_path.stem}.json"
    out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

def load_prediction(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
