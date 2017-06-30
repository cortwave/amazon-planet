from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import numpy as np
import pandas as pd
from itertools import chain
from tqdm import tqdm


class ImageGenerator():
    def __init__(self):
        self.datagen = ImageDataGenerator(
	    horizontal_flip=True,
            vertical_flip=True)

    def get_train_generator(self, x, y, batch_size=128):
        return self.datagen.flow(x, y, batch_size=batch_size)


def get_jpeg_data_files_paths():
    """
    Returns the input file folders path

    :return: list of strings
        The input file paths as list [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]
    """

    data_root_folder = os.path.abspath("../data/")
    train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
    test_jpeg_additional = os.path.join(data_root_folder, 'test-jpg-additional')
    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
    return {"train": train_jpeg_dir,
            "test": test_jpeg_dir,
            "test_add": test_jpeg_additional,
            "train_y": train_csv_file}


def load_image(path, img_size):
    img = Image.open(path)
    img.thumbnail(img_size)
    img_array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    return img_array


def get_train_matrices(train_csv_path, train_path, img_size):
    x = []
    y = []
    train_df = pd.read_csv(train_csv_path)
    labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in train_df['tags'].values])))
    labels_map = {l: i for i, l in enumerate(labels)}
    for file_name, tags in tqdm(train_df.values):
        img = load_image("{}/{}.jpg".format(train_path, file_name), img_size)
        targets = tags_to_vec(tags, labels_map)
        x.append(img)
        y.append(targets)
    return np.asarray(x), np.asarray(y)


def tags_to_vec(labels, labels_map):
    targets = np.zeros(len(labels_map))
    for t in labels.split(' '):
        targets[labels_map[t]] = 1
    return np.asarray(targets)
