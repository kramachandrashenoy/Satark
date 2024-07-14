import os
import shutil
from random import sample, shuffle

def create_subset_dataset(src_dir, dst_dir, num_images_per_class, num_test_images):
    # Create destination directories
    os.makedirs(dst_dir, exist_ok=True)
    train_dst_dir = os.path.join(dst_dir, 'train')
    test_dst_dir = os.path.join(dst_dir, 'test')
    os.makedirs(train_dst_dir, exist_ok=True)
    os.makedirs(test_dst_dir, exist_ok=True)

    # Process train subdirectories
    for class_name in os.listdir(os.path.join(src_dir, 'train')):
        class_dir = os.path.join(src_dir, 'train', class_name)
        if os.path.isdir(class_dir):
            class_dst_dir = os.path.join(train_dst_dir, class_name)
            os.makedirs(class_dst_dir, exist_ok=True)

            # Select a random subset of images
            images = os.listdir(class_dir)
            selected_images = sample(images, min(num_images_per_class, len(images)))

            # Copy selected images to the new directory
            for img in selected_images:
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(class_dst_dir, img)
                shutil.copy(src_path, dst_path)
    
    # Process test directory
    test_src_dir = os.path.join(src_dir, 'test')
    test_images = os.listdir(test_src_dir)
    
    # Shuffle the list of test images
    shuffle(test_images)
    
    # Select a subset of test images
    selected_test_images = test_images[:min(num_test_images, len(test_images))]

    for img in selected_test_images:
        src_path = os.path.join(test_src_dir, img)
        dst_path = os.path.join(test_dst_dir, img)
        shutil.copy(src_path, dst_path)

    print("Dataset subset created successfully.")

# Source and destination directories
source_directory = r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\dataset\imgs"
destination_directory = r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\dataset\new images latest"

# Number of images per class for training and total number of test images
num_images_per_class = 100
num_test_images = 2000

create_subset_dataset(source_directory, destination_directory, num_images_per_class, num_test_images)
