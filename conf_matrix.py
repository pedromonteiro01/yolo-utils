# Author: Pedro Monteiro
# Date: November 2023
# Computer Science Engineering MSc
# Aveiro University

import os
from pathlib import Path
import subprocess

def generate_confusion_matrix(yolov5_dir, data_yaml, weights, batch_size, img_size, conf_thres, iou_thres, task):
    # convert provided directory paths to absolute paths
    yolov5_dir = Path(yolov5_dir).resolve()
    data_yaml = Path(data_yaml).resolve()
    weights = Path(weights).resolve()

    cmd = [
        'python3', 'val.py',
        '--weights', str(weights), # specify the path to the model weights
        '--data', str(data_yaml),  # specify the dataset configuration file
        '--img', str(img_size), # specify the image size
        '--batch', str(batch_size), # specify the batch size
        '--conf', str(conf_thres), # specify the confidence threshold
        '--iou', str(iou_thres), # specify the IoU (Intersection over Union) threshold
        '--task', task, # specify the task (e.g., 'val' for validation)
        '--name', 'conf_matrix'  # name of the run for saving results
    ]

    os.chdir(yolov5_dir) # change the current working directory to YOLOv5

    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True) # execute command

    print(result.stdout)

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    yolov5_dir = script_dir / 'yolov5'
    data_yaml = script_dir / 'dataset.yaml'
    weights = script_dir / 'yolov5' / 'runs' / 'train' / 'exp' / 'weights' / 'best.pt'
    batch_size = 16
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    task = 'val'

    generate_confusion_matrix(yolov5_dir, data_yaml, weights, batch_size, img_size, conf_thres, iou_thres, task)
