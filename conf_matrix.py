import subprocess
import os
from pathlib import Path

def generate_confusion_matrix(yolov5_dir, data_yaml, weights, batch_size, img_size, conf_thres, iou_thres, task):
    # Run the validation script from YOLOv5
    cmd = [
        'python3', 'val.py',
        '--weights', weights,
        '--data', data_yaml,
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--conf', str(conf_thres),
        '--iou', str(iou_thres),
        '--task', task,
        '--name', 'conf_matrix'  # name of the run for saving results
    ]

    # Change the current working directory to YOLOv5
    os.chdir(yolov5_dir)

    # Execute the command
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)

    # Process result if necessary
    print(result.stdout)

# Example usage of the function
if __name__ == "__main__":
    yolov5_dir = base_dir = Path(__file__).parent / 'yolov5'
    data_yaml = base_dir = Path(__file__).parent / 'dataset.yaml'
    weights = base_dir = Path(__file__).parent / 'yolov5s-model.pt'
    batch_size = 16
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    task = 'val'

    generate_confusion_matrix(yolov5_dir, data_yaml, weights, batch_size, img_size, conf_thres, iou_thres, task)