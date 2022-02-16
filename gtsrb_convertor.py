import os
import cv2
import numpy as np
from tqdm import tqdm
import csv


# Transform raw image into numpy format by
# resizing and cropping to the appropriate
# size and then rearranging the dimensions
# of the data to match PyTorch format and
# by transforming the data from 0..255 to 0..1
def numpify_image(image, size=50):
    # Find the smallest resolution dimension and resize
    # until that side becomes size pixels wide
    shape = image.shape
    min_index = 0 if shape[0] < shape[1] else 1
    dim = [0, 0]
    dim[1 - min_index] = size
    dim[min_index] = shape[1 - min_index] * size // shape[min_index]
    resized = cv2.resize(image, tuple(dim))

    # Crop from the longer side the same amount from each side
    # (top and bottom or left and right)
    shape = resized.shape
    amount_to_crop = (shape[1 - min_index] - shape[min_index]) // 2
    crop = (min_index * amount_to_crop, (1 - min_index) * amount_to_crop)

    cropped = resized[crop[0]:size + crop[0], crop[1]:size + crop[1]]
    # Return in correct order of dimensions and normalised to 0..1
    return np.transpose(cropped, (2, 0, 1)) / 255.


# Turns raw GTSRB training images into numpy format
def convert_training_data():
    print("Converting training data to numpy")

    train_path = "Data/GTSRB/GTSRB_Final_Training_Images/" \
                 "GTSRB/Final_Training/Images"
    data = []

    # For every folder i.e. class append the
    # transformed images alongside their labels in tuples
    print("Loading and converting training data...")
    for label_name in tqdm(os.listdir(train_path)):
        label = int(label_name)
        image_path = f"{train_path}/{label_name}"
        for image_name in os.listdir(image_path):
            if ".ppm" not in image_name:
                continue
            img = numpify_image(cv2.cvtColor(
                cv2.imread(
                    os.path.join(image_path, image_name),
                    cv2.IMREAD_COLOR),
                cv2.COLOR_RGB2BGR))
            data.append((img.tolist(), label))

    # Shuffle the tuples and extract the lists of
    # first elements and second elements of tuples,
    # images and labels, respectively
    print("Shuffling training data...")
    np.random.shuffle(data)
    images, labels = list(zip(*data))

    print("Saving training data...")
    np.save("Data/GTSRB/train_images", images)
    np.save("Data/GTSRB/train_labels", labels)


# Turns raw GTSRB testing images into numpy format
def convert_testing_data():
    print("Converting testing data to numpy")

    images_path = "Data/GTSRB/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    labels_path = "Data/GTSRB/GTSRB_Final_Test_GT/GT-final_test.csv"

    # Extract labels from csv file and remove header
    with open(labels_path, newline="") as f:
        reader = csv.reader(f)
        names = dict(reader)
        names.pop("Filename", None)

    # Cast the ClassId column from string to int
    for key in names:
        names[key] = int(names[key])

    images = []
    labels = []

    print("Loading testing data...")
    for image_name in tqdm(os.listdir(images_path)):
        if ".ppm" not in image_name:
            continue
        img = numpify_image(cv2.cvtColor(
            cv2.imread(
                os.path.join(images_path, image_name),
                cv2.IMREAD_COLOR),
            cv2.COLOR_RGB2BGR))
        images.append(img.tolist())
        labels.append(names[image_name])

    print("Saving testing data...")
    np.save("Data/GTSRB/test_images", images)
    np.save("Data/GTSRB/test_labels", labels)


# Will convert all the data successively
if __name__ == "__main__":
    convert_training_data()
    convert_testing_data()
