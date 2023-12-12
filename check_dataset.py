import os

image_folder_path = 'dataset/valid/images'
txt_folder_path = 'dataset/valid/labels'

image_filenames = {os.path.splitext(file)[0] for file in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, file))}

txt_filenames = {os.path.splitext(file)[0] for file in os.listdir(txt_folder_path) if file.endswith('.txt')}

# compare the two sets
images_without_txt = image_filenames - txt_filenames
txts_without_images = txt_filenames - image_filenames

if not images_without_txt and not txts_without_images:
    print("All image filenames have corresponding .txt files, and vice versa.")
else:
    if images_without_txt:
        print("Images without corresponding .txt files:", images_without_txt)
    if txts_without_images:
        print("Txt files without corresponding images:", txts_without_images)
