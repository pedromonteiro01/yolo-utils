import os
from pathlib import Path
import subprocess

def generate_confusion_matrix(yolov5_dir, data_yaml, weights, batch_size, img_size, conf_thres, iou_thres, task):
    # Ensure the paths are absolute
    yolov5_dir = Path(yolov5_dir).resolve()
    data_yaml = Path(data_yaml).resolve()
    weights = Path(weights).resolve()

    # Run the validation script from YOLOv5
    cmd = [
        'python3', 'val.py',
        '--weights', str(weights),
        '--data', str(data_yaml),
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
