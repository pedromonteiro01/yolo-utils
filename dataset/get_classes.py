import os

def filter_labels(labels_folder, keep_classes):
    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(file_path, 'w') as file:
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id in keep_classes:
                        file.write(line)

# define the classes to keep
keep_classes = {0, 2}  # indices for 'pistol' and 'knife'

labels_folder = 'train/labels'

filter_labels(labels_folder, keep_classes)