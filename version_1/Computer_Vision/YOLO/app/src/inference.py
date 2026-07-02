from ultralytics import YOLO
from pathlib import Path
import torch
from PIL import Image
import numpy
from src.config import load_config
class YOLOv11Inference:
    def __init__(self,model_name,device="cuda"):
        self.model=YOLO(model_name)
        self.device=device
        config=load_config()
        self.conf_threshold=config["model"]["conf_threshold"]
        self.image_extension=config["data"]["image_extension"]
    def process_image(self,image_path):
        self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device
        )

        detections=[]
        class_counts={}
        for result in results:
            for box in result.boxes:
                cls=result.names[int(box.cls)]
                conf=float(box.conf)
                bbox=box.xyxy[0].tolist()
                detections.append({
                    'class':cls,
                    'confidence':conf,
                    'bbox':bbox,
                    'count':1
                })
                class_counts[cls]=class_counts.get(cls,0)+1
  
        
    def process_directory(self,directory):
        metdata=[]
        patterns=[f"*{ext}" for ext in self.image_extension]
        image_paths=[]
        for pattern in patterns:
            image_paths.extend(Path(directory).glob(pattern))
        
        for img_path in image_paths:
            try:
                metdata.extend(self.process_image(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue