import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml

def evaluate_model(weight_file):
    command = f"python3 val.py --weights {weight_file} --data dataset.yaml --img 640 --conf 0.25 --iou 0.5"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    
    return result.stderr

def extract_metrics(output):
    mAP50 = 0.0
    
    for line in output.split("\n"):
        if "all" in line:
            metrics = line.split()
            mAP50 = float(metrics[5])
    
    return mAP50

def extract_details_from_opt(exp_path):
    with open(os.path.join(exp_path, 'opt.yaml'), 'r') as f:
        opt_data = yaml.safe_load(f)
        epochs = opt_data['epochs']
        batch_size = opt_data['batch_size']
    return epochs, batch_size

def find_best_config():
    base_dir = "yolov5/runs/train"
    
    best_weight = None
    best_mAP50 = 0.0
    best_epochs = 0
    best_batch_size = 0
    
    # lists to store values for plotting
    mAP50_values = []
    all_epochs = []
    all_batch_sizes = []

    # iterate through each exp directory
    for exp_dir in os.listdir(base_dir):
        if exp_dir.startswith("exp"):
            current_dir = os.path.join(base_dir, exp_dir, "weights")
            if not os.path.exists(current_dir):
                continue
            
            epochs, batch_size = extract_details_from_opt(os.path.join(base_dir, exp_dir)) # get details from opt.yaml

            print(f"Checking weight files in {current_dir}:")
            print(f"epochs: {epochs} batch size: {batch_size}")

            for filename in os.listdir(current_dir): # iterate through weight files in the current exp directory
                if filename.endswith(".pt"):
                    weight_path = os.path.join(current_dir, filename)

                    output = evaluate_model(weight_path)
                    mAP50 = extract_metrics(output)  # Extract mAP50 metric
                    
                    # store plot values
                    mAP50_values.append(mAP50)
                    all_epochs.append(epochs)
                    all_batch_sizes.append(batch_size)

                    if mAP50 > best_mAP50:
                        best_mAP50 = mAP50
                        best_weight = weight_path
                        best_epochs = epochs
                        best_batch_size = batch_size
    
    print(f"\n\nbest_weight {best_weight}\n")
    print(f"best_mAP50 {best_mAP50}\n")
    print(f"best_epochs {best_epochs}\n")
    print(f"best_batch_size {best_batch_size}\n")
    print(f"mAP50_values {mAP50_values}\n")
    print(f"all_epochs {all_epochs}\n")
    print(f"all_batch_sizes {all_batch_sizes}\n\n")
                        
    return best_weight, best_mAP50, best_epochs, best_batch_size, mAP50_values, all_epochs, all_batch_sizes

def plot_results(mAP50_values, epochs, batch_sizes):
    plt.figure(figsize=(10, 6))
    
    # mAP50 vs. Epochs
    scatter = plt.scatter(epochs, mAP50_values, c=batch_sizes, cmap='viridis', s=100, edgecolors='k')
    plt.title('mAP50 vs. Epochs (Color-coded by Batch Size)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP50')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Batch Size')
    
    plt.tight_layout()
    plt.savefig("mAP50_results.png")

if __name__ == "__main__":
    best_weight, best_mAP50, best_epochs, best_batch_size, all_mAP50_values, all_epochs, all_batch_sizes = find_best_config()
    print(f"Best weight file: {best_weight} with mAP50: {best_mAP50}")
    print(f"Number of Epochs: {best_epochs}, Batch Size: {best_batch_size}")
    plot_results(all_mAP50_values, all_epochs, all_batch_sizes)
