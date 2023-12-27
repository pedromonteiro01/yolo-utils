# Author: Pedro Monteiro
# Date: November 2023
# Computer Science Engineering MSc
# Aveiro University

import os
import torch
import argparse
from yolov5.utils.metrics import bbox_iou
import matplotlib.pyplot as plt
import re

def load_tensor_from_file(filename):
    # load and convert bounding box data from a file into a PyTorch tensor
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = [list(map(float, line.strip().split())) for line in lines]
    return torch.tensor(data)

def get_sample_data(folder_path):
    # retrieve bounding box data from all text files in a folder
    files = [load_tensor_from_file(os.path.join(folder_path, file))
             for file in os.listdir(folder_path) if file.endswith('.txt')]
    return files

def yolo_to_corner(box):
    # convert YOLO format bounding boxes to corner format
    x_center, y_center, width, height = box
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)
    return torch.tensor([x1, y1, x2, y2])

def compute_average_iou(detections, labels):
    # calculate the average IoU between detections and labels for a dataset
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
        print("No IoU values computed for folder. There might be no detections or labels.")
    return sum(iou_values) / len(iou_values) if iou_values else None

def main(detect_folder, valid_folder, img_sizes):
    labels = get_sample_data(valid_folder)
    results = []

    for folder_name in os.listdir(detect_folder):
        if 'batch' in folder_name and 'epoch' in folder_name and 'img' in folder_name:
            folder_path = os.path.join(detect_folder, folder_name)
            batch_size, epoch_number, img_size = extract_parameters(folder_name)

            if img_size in img_sizes:
                detect_labels_path = os.path.join(folder_path, 'labels')
                
                if os.path.isdir(detect_labels_path) and any(fname.endswith('.txt') for fname in os.listdir(detect_labels_path)):
                    detections = get_sample_data(detect_labels_path)
                    avg_iou = compute_average_iou(detections, labels)
                    if avg_iou is not None:
                        results.append((img_size, batch_size, epoch_number, avg_iou))
                    else:
                        print(f"No valid IoU for: {folder_name}")
                else:
                    print(f"No label files in: {detect_labels_path}")
            else:
                print(f"Skipping {folder_name} as it is not part of the specified image sizes.")
        else:
            print(f"Skipping {folder_name} as it does not match the expected pattern.")

    if not results:
        print("No IoU results to plot.")
    else:
        plot_iou_results(results)

def extract_parameters(folder_name):
    batch_match = re.search(r'batch(\d+)', folder_name)
    epoch_match = re.search(r'epoch(\d+)', folder_name)
    imgsize_match = re.search(r'img(\d+)', folder_name)
    
    batch_size = int(batch_match.group(1)) if batch_match else None
    epoch_number = int(epoch_match.group(1)) if epoch_match else None
    img_size = int(imgsize_match.group(1)) if imgsize_match else None

    return batch_size, epoch_number, img_size


def plot_iou_results(results):
    if not results:
        print("No IoU results to plot.")
        return
    
    # plot the IoU results against epoch numbers, batch sizes, and image sizes
    batch_sizes = [result[1] for result in results]
    epoch_numbers = [result[2] for result in results]
    img_sizes = [result[0] for result in results]
    iou_values = [result[3] for result in results]

    max_batch_size = max(batch_sizes)
    min_batch_size = min(batch_sizes)
    batch_size_range = max_batch_size - min_batch_size
    batch_sizes_normalized = [(50 if batch_size_range == 0 else 
                               (bs - min_batch_size) / batch_size_range * 100 + 10) 
                              for bs in batch_sizes]

    plt.figure(figsize=(10, 6))
    
    unique_img_sizes = list(set(img_sizes))
    img_size_to_color = {size: i for i, size in enumerate(unique_img_sizes)}
    colors = [img_size_to_color[size] for size in img_sizes]

    scatter = plt.scatter(epoch_numbers, iou_values, c=colors, s=batch_sizes_normalized, cmap='viridis', edgecolors='k')
    plt.title('Average IoU vs. Epoch Number (Color-coded by Image Size, Size-coded by Batch Size)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Average IoU')
    
    cbar = plt.colorbar(scatter, ticks=range(len(unique_img_sizes)))
    cbar.ax.set_yticklabels([str(size) for size in unique_img_sizes])
    cbar.set_label('Image Size')

    if len(set(batch_sizes)) > 1:
        handles, labels = [], []
        for bs in sorted(set(batch_sizes)):
            handles.append(plt.scatter([], [], s=((bs - min_batch_size) / batch_size_range * 100 + 10 if batch_size_range else 50), color='grey', alpha=0.7))
            labels.append(f'Batch Size: {bs}')
        plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), scatterpoints=1, frameon=False, labelspacing=1, title='Batch Size', ncol=len(handles))

    plt.tight_layout()
    plt.savefig("iou_results_by_imgsize_and_batch.png")


if __name__ == '__main__':
    main()
