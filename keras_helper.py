from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from keras.models import Model
import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend
from keras.applications.resnet50 import ResNet50

from PIL import Image
from tqdm import tqdm
import numpy as np

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2] * 17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
            x[i] = best_i2
        if verbose:
           print(i, best_i2, best_score)

    return x

 
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class Fbeta(Callback):
    def __init__(self, valid_data):
        super().__init__()
        self.fbeta = []
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs ={}):
        p_valid = self.model.predict(self.validation_data[0])
        y_val = self.validation_data[1][0]
        thresholds = optimise_f2_thresholds(y_val, p_valid, verbose=False)
        f_beta = fbeta_score(y_val, np.array(p_valid) > thresholds, beta=2, average='samples')
        print("fbeta_score = {}".format(f_beta))
        self.fbeta.append(f_beta)
        return


class KerasModel:
    def __init__(self, img_size, img_channels=3, output_size=17):
        self.losses = []
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(img_size[0], img_size[1], img_channels)))

        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(output_size, activation='sigmoid'))

    def get_fbeta_score(self, validation_data, verbose=True):
        p_valid = self.model.predict(validation_data[0])
        thresholds = optimise_f2_thresholds(validation_data[1], p_valid, verbose=verbose)
        return fbeta_score(validation_data[1], np.array(p_valid) > thresholds, beta=2, average='samples'), thresholds

    def fit(self, flow, epochs, lr, validation_data, train_callbacks=[], batches=300):
        history = LossHistory()
        fbeta = Fbeta(validation_data)
        opt = Adam(lr=lr)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        self.model.fit_generator(flow, steps_per_epoch=batches, epochs=epochs, callbacks=[history, earlyStopping, fbeta] + train_callbacks, validation_data=validation_data)
        fb_score, thresholds = self.get_fbeta_score(validation_data, verbose=False)
        return [fbeta.fbeta, history.train_losses, history.val_losses, fb_score, thresholds]

    def save_weights(self, weight_file_path):
        self.model.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def predict_image(self, image):
        img = Image.fromarray(np.uint8(image * 255))
        images = [img.copy().rotate(i) for i in [-90, 90, 180]]
        images.append(img)
        images = np.asarray([np.asarray(image.convert("RGB"), dtype=np.float32) / 255 for image in images])
        return sum(self.model.predict(images)) / 4

    def predict(self, x_test):
        return [self.predict_image(img) for img in tqdm(x_test)]

    def map_predictions(self, predictions, labels_map, thresholds):
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)
        return predictions_labels

    def close(self):
        backend.clear_session()
