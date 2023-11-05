import os
import torch
import argparse
from yolov5.utils.metrics import bbox_iou

def load_tensor_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [list(map(float, line.strip().split())) for line in lines]
    return torch.tensor(data)

def get_sample_data(folder_path):
    files = [load_tensor_from_file(os.path.join(folder_path, file))
             for file in os.listdir(folder_path) if file.endswith('.txt')]
    return files

def yolo_to_corner(box):
    x_center, y_center, width, height = box
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)
    return torch.tensor([x1, y1, x2, y2])

def compute_average_iou(detections, labels):
    iou_values = []
    if not detections or not labels:
        return None
    for det_image, lab_image in zip(detections, labels):
        if det_image.size(0) == 0 or lab_image.size(0) == 0:
            print("No detections or labels for an image.")
            continue
        for det in det_image:
            det_box = yolo_to_corner(det[1:5]).unsqueeze(0)
            max_iou = 0
            for lab in lab_image:
                lab_box = yolo_to_corner(lab[1:5]).unsqueeze(0)
                iou = bbox_iou(det_box, lab_box, xywh=False).item()
                if iou > max_iou:
                    max_iou = iou
            iou_values.append(max_iou)
    if not iou_values:
        print("No IoU values computed. There might be no detections or labels.")
        return None
    return sum(iou_values) / len(iou_values)


def main(detect_folder=None, valid_folder=None):
    if detect_folder is None or valid_folder is None:
        parser = argparse.ArgumentParser(description='IoU Calculation')
        parser.add_argument('--detect_folder', required=True, help='Path to detection folders')
        parser.add_argument('--valid_folder', required=True, help='Path to validation label files')
        args = parser.parse_args()
        detect_folder = args.detect_folder
        valid_folder = args.valid_folder

    # Load validation labels once as they are the same for all experiments
    labels = get_sample_data(valid_folder)

    # Store IoU results along with experiment folder names
    results = []

    # Iterate over all folders under the specified detect_folder
    for folder_name in os.listdir(detect_folder):
        folder_path = os.path.join(detect_folder, folder_name)
        # Check if the folder name matches the pattern "batch*_epoch*"
        if os.path.isdir(folder_path) and 'batch' in folder_name and 'epoch' in folder_name:
            detect_labels_path = os.path.join(folder_path, 'labels')
            detections = get_sample_data(detect_labels_path)
            avg_iou = compute_average_iou(detections, labels)
            if avg_iou is not None:
                results.append((folder_path, avg_iou))

    # Sort and display results by descending IoU
    results.sort(key=lambda x: x[1], reverse=True)
    for folder_path, avg_iou in results:
        print(f"Evaluating folder: {folder_path}")
        print(f"Average IoU for {folder_path}: {avg_iou:.2f}\n")

if __name__ == '__main__':
    main()
