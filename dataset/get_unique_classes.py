import os

def count_classes_in_yolo_labels(labels_folder):
    unique_classes = set()

    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)

            with open(file_path, 'r') as file:
                for line in file:
                    class_id = int(line.split()[0]) # extract the class ID and add to the set
                    unique_classes.add(class_id)

    return unique_classes

labels_folder = 'train/labels/'
classes = count_classes_in_yolo_labels(labels_folder)
print(f"Number of unique classes: {len(classes)}")
print(f"Unique classes: {classes}")
