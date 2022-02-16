import os
import cv2
import numpy as np
from tqdm import tqdm
import sys


# Transform raw image into numpy format by rearranging the
# dimensions of the data to match PyTorch format
# and by transforming the data from 0..255 to 0..1
def numpify_image(image):
    return np.transpose(image, (2, 0, 1)) / 255.


# Since loading all 100000 training images
# causes the program to crash, the data must
# be split up into small chunks
# The most effective way to do this is assign
# a random name to an image, from 0 to 99999,
# and move them all into another folder where
# they will be sorted alpha-numerically. Now
# shuffled offline, they can easily be converted
# into numpy format without affecting the
# training process.
def randomize_and_move_images():
    print("Randomizing and renaming data...")

    train_path = "Data/TinyImageNet/tiny-imagenet-200/tiny-imagenet-200/train"
    move_path = "Data/TinyImageNet/tiny-imagenet-200/tiny-imagenet-200/train/images"
    if not os.path.exists(move_path):
        os.makedirs(move_path)
    # Shuffle the indices/names of the future values
    file_names = np.arange(100000)
    np.random.shuffle(file_names)

    class_counter = 0
    label_counter = 0
    classes = {}
    labels = {}

    # For each class, load the files and rename them
    for folder in tqdm(os.listdir(train_path)):
        # Passes over any file or directory without
        # the letter "n" in its name as all class
        # directories are named
        # something like n<number>.
        if "n" not in folder:
            continue
        # On first accessing a folder, remember
        # its corresponding numeric class for later
        classes[folder] = class_counter
        label = class_counter
        class_counter += 1

        # Keep track of what label each new image name has
        image_path = f"{train_path}/{folder}/images"
        for image_name in os.listdir(image_path):
            new_image_name = file_names[label_counter]
            label_counter += 1
            new_image_name = str(new_image_name).rjust(5, "0") + ".JPEG"
            os.rename(f"{image_path}/{image_name}", f"{move_path}/{new_image_name}")
            labels[new_image_name] = label

    # Log the sorted labels dictionary for the loading training set step
    with open(f"{train_path}/labels.txt", "w") as f:
        for key in sorted(labels):
            f.write(f"{key} {labels[key]}\n")

    # Log the classes dictionary for the loading testing set step
    with open("Data/TinyImageNet/classes.txt", "w") as f:
        for key in classes:
            f.write(f"{key} {classes[key]}\n")


# Loads the randomised data from the
# output of the above function, and
# saves the transformed into numpy format
# mega-batches
def convert_training_data():
    print("Converting training data to numpy")
    # 4 mega batches seemed the most appropriate trade-off
    memory_batches = 4
    images_per_batch = 100000 // memory_batches

    images_path = "Data/TinyImageNet/tiny-imagenet-200/" \
                  "tiny-imagenet-200/train/images"
    labels_path = "Data/TinyImageNet/tiny-imagenet-200/" \
                  "tiny-imagenet-200/train/labels.txt"
    with open(labels_path, "r") as f:
        names = dict([tuple(line.strip().split(" ")) for line in f])

    # Cast the loaded label names into int
    for key in names:
        names[key] = int(names[key])

    images = []
    labels = []
    batch = 0
    current_image_count = 0

    print("Loading...")
    for image in tqdm(os.listdir(images_path)):
        img = numpify_image(cv2.cvtColor(
            cv2.imread(
                os.path.join(images_path, image),
                cv2.IMREAD_COLOR),
            cv2.COLOR_RGB2BGR))
        images.append(img.tolist())
        labels.append(names[image])
        current_image_count += 1

        # If enough images have been loaded to
        # fill out a batch, then save the batch
        # and start a new one
        if current_image_count == images_per_batch:
            print(f"Saving batch {batch}...")
            np.save(f"Data/TinyImageNet/train_images_{batch}", images)
            np.save(f"Data/TinyImageNet/train_labels_{batch}", labels)
            current_image_count = 0
            batch += 1
            images = []
            labels = []


# Loads the test images and converts into numpy format
def convert_testing_data():
    print("Converting testing data to numpy")

    images_path = "Data/TinyImageNet/tiny-imagenet-200/" \
                  "tiny-imagenet-200/val/images"
    labels_path = "Data/TinyImageNet/tiny-imagenet-200/" \
                  "tiny-imagenet-200/val/val_annotations.txt"
    classes_path = "Data/TinyImageNet/classes.txt"

    with open(labels_path, "r") as f:
        names = dict([tuple(line.strip().split("\t")[0:2]) for line in f])

    with open(classes_path, "r") as f:
        classes = dict([tuple(line.strip().split(" ")) for line in f])

    for key in classes:
        classes[key] = int(classes[key])

    images = []
    labels = []

    print("Loading...")
    for image_name in tqdm(os.listdir(images_path)):
        img = numpify_image(cv2.cvtColor(
            cv2.imread(
                os.path.join(images_path, image_name),
                cv2.IMREAD_COLOR),
            cv2.COLOR_RGB2BGR))
        images.append(img.tolist())
        labels.append(classes[names[image_name]])

    print("Saving...")
    np.save("Data/TinyImageNet/test_images", images)
    np.save("Data/TinyImageNet/test_labels", labels)


# Completes the full TinyImageNet conversion in the recommended (only) order.
# This is the function that must be run, not any other one
def convert_data():
    randomize_and_move_images()
    convert_training_data()
    convert_testing_data()


# Converts the Tiny-ImageNet-C images to numpy
# and saves the resulting files in the same
# format as the CIFAR-10-C and CIFAR-100-C data sets
def convert_tinyimagenet_c():
    print("Converting data to numpy...")

    data_path = "Data/Tiny-ImageNet-C/Tiny-ImageNet-C"

    # Across all severities, the images are organised
    # in the same way: 50 images
    labels = []
    for _ in range(5):
        for label in range(200):
            labels.extend([label] * 50)

    for corruption in tqdm(os.listdir(data_path)):
        if ".npy" in corruption:
            continue
        corruption_path = os.path.join(data_path, corruption)
        images = []
        for severity in os.listdir(corruption_path):
            severity_path = os.path.join(corruption_path, severity)
            for folder in os.listdir(severity_path):
                images_path = os.path.join(severity_path, folder)
                for image in os.listdir(images_path):
                    images.append(cv2.cvtColor(
                        cv2.imread(
                            os.path.join(images_path, image),
                            cv2.IMREAD_COLOR),
                        cv2.COLOR_RGB2BGR))
        np.save(f"{data_path}/{corruption}", images)
    np.save(f"{data_path}/labels", labels)


if __name__ == "__main__":
    # First boolean value converts the original data set
    # Second value converts the Tiny-ImageNet-C data set
    convert_tinyimagenet = False
    convert_tinyimagenetc = True

    # Logic for running code from command line
    if len(sys.argv) > 3:
        print("Too many arguments!")
        sys.exit(1)
    elif len(sys.argv) == 3:
        if sys.argv[1] == "-d" or sys.argv[2] == "-d":
            convert_tinyimagenet = True
        if sys.argv[1] == "-c" or sys.argv[2] == "-c":
            convert_tinyimagenetc = True
    elif len(sys.argv) == 2:
        if sys.argv[1] == "-d":
            convert_tinyimagenet = True
        if sys.argv[1] == "-c":
            convert_tinyimagenetc = True

    if convert_tinyimagenet:
        randomize_and_move_images()
        convert_training_data()
        convert_testing_data()
    if convert_tinyimagenetc:
        convert_tinyimagenet_c()

