# Author: Pedro Monteiro
# Date: November 2023
# Computer Science Engineering MSc
# Aveiro University

import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import numpy as np

def evaluate_model(weight_file, img_size):
    command = f"python3 yolov5/val.py --weights {weight_file} --data dataset.yaml --img {img_size} --conf 0.25 --iou 0.5"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    
    return result.stderr

def extract_metrics(output):
    mAP50 = 0.0 # initialize mAP50 metric to zero
    
    for line in output.split("\n"): # parse the output to extract mAP50 value
        if "all" in line:
            metrics = line.split()
            mAP50 = float(metrics[5]) # extract mAP50 from the parsed line
    
    return mAP50

def extract_details_from_opt(exp_path):
    # read opt.yaml file in the experiment path to extract training details
    with open(os.path.join(exp_path, 'opt.yaml'), 'r') as f:
        opt_data = yaml.safe_load(f)
        epochs = opt_data['epochs']
        batch_size = opt_data['batch_size']
    return epochs, batch_size

def find_best_config(img_sizes):
    base_dir = "yolov5/runs/train"
    
    # initialize variables to track the best model configuration
    best_weight = None
    best_mAP50 = 0.0
    best_epochs = 0
    best_batch_size = 0
    best_img_size = 0
    
    # lists to store values for plotting
    mAP50_values = []
    all_epochs = []
    all_batch_sizes = []
    all_img_sizes = []

    # iterate through each exp directory
    for exp_dir in os.listdir(base_dir):
        if exp_dir.startswith("exp"):
            current_dir = os.path.join(base_dir, exp_dir, "weights")
            if not os.path.exists(current_dir):
                continue
            
            epochs, batch_size = extract_details_from_opt(os.path.join(base_dir, exp_dir)) # get details from opt.yaml

            print(f"Checking weight files in {current_dir}:")
            print(f"epochs: {epochs} batch size: {batch_size}")

            for img_size in img_sizes:
                for filename in os.listdir(current_dir): # iterate through weight files in the current exp directory
                    if filename.endswith(".pt"):
                        weight_path = os.path.join(current_dir, filename)

                        output = evaluate_model(weight_path, img_size)
                        mAP50 = extract_metrics(output)  # extract mAP50 metric
                        
                        # store plot values
                        mAP50_values.append(mAP50)
                        all_epochs.append(epochs)
                        all_batch_sizes.append(batch_size)
                        all_img_sizes.append(img_size)

                        if mAP50 > best_mAP50: # update the best model configuration if current mAP50 is higher
                            best_mAP50 = mAP50
                            best_weight = weight_path
                            best_epochs = epochs
                            best_batch_size = batch_size
                            best_img_size = img_size
                        
    return best_weight, best_mAP50, best_epochs, best_batch_size, mAP50_values, all_epochs, all_batch_sizes, all_img_sizes


def plot_results(mAP50_values, epochs, batch_sizes, img_sizes):
    plt.figure(figsize=(10, 6))
    
    max_batch_size = max(batch_sizes)
    min_batch_size = min(batch_sizes)
    batch_size_range = max_batch_size - min_batch_size
    batch_sizes_normalized = [
        50 if batch_size_range == 0 else (bs - min_batch_size) / batch_size_range * 100 + 10 
        for bs in batch_sizes
    ]

    unique_img_sizes = list(set(img_sizes))
    img_size_to_color = {size: i for i, size in enumerate(unique_img_sizes)}
    colors = [img_size_to_color[size] for size in img_sizes]

    scatter = plt.scatter(epochs, mAP50_values, c=colors, s=batch_sizes_normalized, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title('mAP50 vs. Epochs (Color-coded by Image Size, Size-coded by Batch Size)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP50')

    cbar = plt.colorbar(scatter, ticks=range(len(unique_img_sizes)), pad=0.1)
    cbar.ax.set_yticklabels([str(size) for size in unique_img_sizes])
    cbar.set_label('Image Size')

    if len(set(batch_sizes)) > 1:
        batch_size_handles = [
            plt.scatter([], [], s=(bs - min_batch_size + 1) * 10, label=f'Batch Size: {bs}', color='gray', alpha=0.6)
            for bs in sorted(set(batch_sizes))
        ]
        batch_legend = plt.legend(handles=batch_size_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, title='Batch Sizes')
        plt.gca().add_artist(batch_legend)

    plt.tight_layout(rect=[0, 0.1, 0.9, 0.9]) 
    plt.savefig("mAP50_results_by_img_size_and_batch_size.png")
    
if __name__ == "__main__":
    best_weight, best_mAP50, best_epochs, best_batch_size, all_mAP50_values, all_epochs, all_batch_sizes, all_img_sizes = find_best_config([256, 640])  # Replace with actual img_sizes if different
    print(f"Best weight file: {best_weight} with mAP50: {best_mAP50}")
    print(f"Number of Epochs: {best_epochs}, Batch Size: {best_batch_size}")
    plot_results(all_mAP50_values, all_epochs, all_batch_sizes, all_img_sizes)
