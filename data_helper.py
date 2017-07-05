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
            rescale=1./255,
            zoom_range=0.2,
	    horizontal_flip=True,
            vertical_flip=True)

    def get_train_generator(self, x, y, batch_size=128):
        return self.datagen.flow(x, y, batch_size=batch_size)

class ValidGenerator():
    def __init__(self):
        self.datagen = ImageDataGenerator(
            rescale=1./255)

    def get_valid_generator(self, x, y, batch_size=128):
        return self.datagen.flow(x, y, batch_size=batch_size)




def get_jpeg_data_files_paths():
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
    img_array = np.asarray(img.convert("RGB"), dtype=np.int8)
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
    labels_map = {v: k for k, v in labels_map.items()}
    return np.asarray(x), np.asarray(y), labels_map

def get_test_matrices(test_dir, img_size):
    x_test = []
    x_test_filename = []
    files_name = os.listdir(test_dir)
    for file_name in tqdm(files_name):
        img = load_image("{}/{}".format(test_dir, file_name), img_size)
        x_test.append(img)
        x_test_filename.append(file_name)
    return np.array(x_test), x_test_filename


def tags_to_vec(labels, labels_map):
    targets = np.zeros(len(labels_map))
    for t in labels.split(' '):
        targets[labels_map[t]] = 1
    return np.asarray(targets)
