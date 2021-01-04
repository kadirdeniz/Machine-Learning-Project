import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

IMG_WIDTH = 30
IMG_HEIGHT = 30
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )


def load_data(data_dir):
    images = []
    labels = []
    
    for folder in os.listdir(data_dir):
        if folder == ".DS_Store": continue
        else:
            folder_path = os.path.join(data_dir, folder)
            for img in os.listdir(folder_path):
                image_path = os.path.join(folder_path, img)
                read_img = cv2.imread(image_path)
                resize_img = cv2.resize(read_img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(resize_img)
                labels.append(int(folder))

    return np.array(images), np.array(labels)
    
if __name__ == "__main__":
    main()