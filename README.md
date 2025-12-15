# Computer Vision Railway Validation

This project implements an AI-based validation agent for railway computer vision systems.  
The agent processes railway video footage and runs object detection algorithms to identify obstacles and relevant entities under diverse environmental conditions. Validation focuses on both quantitative performance metrics and qualitative visual analysis, including side-by-side comparison between multiple algorithm versions to identify improvements, regressions, and safety-critical edge cases.

The database I use is **RailGoerl24**, introduced in  
[RailGoerl24: Görlitz Rail Test Center CV Dataset 2024](data.fid-move.de/dataset/railgoerl24)

The annotated dataset comprises **61 video sequences**. These sequences include **12,205 frames** with **33,556 bounding box annotations of persons**. The video sequences were recorded at a frame rate of **25 fps**, and every **15th frame** starting from frame **0** was extracted for annotation purposes. Annotations’ scatter plots are depicted in Fig. 4, showing a broad range of various person sizes as well as person positions.

It was chosen because... 

One of the main goals in the choice of scenarios for RailGoerl24 was achieving representative diversity of recorded human beings for a railway operational design domain in Germany.

Since the purpose of this project is to demonstrate my validation capabilities rather than training a CV model, I curated a subset of the dataset for training to decrease runtime. The subset was selected to ensure coverage of rare but high-risk scenarios such as persons lying on or between rails, in addition to standard pedestrian cases:

- **6** videos of people lying down / fallen (Person Liegend / gestürzt) 
- **3** Person sitting / crouching (Person-hockend) (hard detection)
- **2** difficult lighting (Gegenlicht) 
- **5** people crossing, 
- **7** blurry people and groups, children (Fuzzy-KinderMenschengruppe1.mp4)

Total of **23 videos**
In RailGoerl24, annotations are in **XML** format. The model used for training and inference (YOLO) expects predictions as **one-line text files**. Therefore, in order to do validation and calculate metrics, I need to convert the format of the XML to that of the YOLO output.

## 1️⃣ Small / distant pedestrians

Railway scenes = people often appear:
- very small  
- near the horizon  
- partially occluded  

YOLOv8n misses these more often.

**YOLOv8s:**
- stronger feature extractor  
- better recall  
- fewer false negatives  

So validation requires parsing XML and matching boxes.  
I parse the result of YOLO to parse XML directly, since I want to keep original annotations and not lose dataset-specific metadata (unsure_objects, autoAnnotated, etc.).


```xml
<name>person</name>
<bndbox>
    <xmin>562</xmin>
    <ymin>260</ymin>
    <xmax>584</xmax>
    <ymax>306</ymax>
</bndbox>
This gives  one GT box:
(xmin, ymin, xmax, ymax)
YOLO gives :
(x1, y1, x2, y2, confidence)

The database annotates humans only, so detection of anything else in YOLO (train,truck,car) are ignored.