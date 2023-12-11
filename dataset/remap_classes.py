import os

def update_and_filter_labels(labels_folder, class_mapping):
    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(file_path, 'w') as file:
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id in class_mapping:
                        new_class_id = class_mapping[class_id]
                        updated_line = line.replace(str(class_id), str(new_class_id), 1)
                        file.write(updated_line)

class_mapping = {0: 0, 2: 1}  # map 'pistol' to 0 and 'knife' to 1

labels_folder = 'train/labels' # path to labels

update_and_filter_labels(labels_folder, class_mapping)