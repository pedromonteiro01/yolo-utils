import os
import shutil
import random

# Define the paths for images and labels
images_path = 'data/images'
labels_path = 'data/labels'

# List all files in the images directory
images = os.listdir(images_path)

# Shuffle the list for random splitting
random.shuffle(images)

# Calculate the split index for 80% training data
split_index = int(0.8 * len(images))

# Split the images into training and validation sets
train_images = images[:split_index]
valid_images = images[split_index:]

# Function to move images and their corresponding labels
def move_files(image_list, dataset_type):
    # Create necessary directories
    os.makedirs(f'dataset/{dataset_type}/images', exist_ok=True)
    os.makedirs(f'dataset/{dataset_type}/labels', exist_ok=True)

    # Move each image and its label (if exists) to the respective directory
    for image in image_list:
        shutil.move(os.path.join(images_path, image), f'dataset/{dataset_type}/images/{image}')
        label_file = image.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_path, label_file)
        if os.path.exists(label_path):
            shutil.move(label_path, f'dataset/{dataset_type}/labels/{label_file}')

# Move train and validation images and labels
move_files(train_images, 'train')
move_files(valid_images, 'valid')

print("Dataset organized into train and validation sets.")
