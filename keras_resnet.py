from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
import keras as k
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend
from keras.applications.resnet50 import ResNet50

from PIL import Image
from tqdm import tqdm
import numpy as np


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class ResnetModel:
    def __init__(self, output_size=17):
        self.losses = []
        self.resnet = ResNet50(include_top=False, weights='imagenet')
        x = self.resnet.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(output_size, activation='sigmoid')(x)
        self.model = Model(input=self.resnet.input, output=predictions)

    def get_fbeta_score(self, validation_data, validation_labels, validation_steps, verbose=True):
        p_valid = self.model.predict_generator(validation_data, validation_steps)
        thresholds = self.optimise_f2_thresholds(validation_labels, p_valid, verbose=verbose)
        return fbeta_score(validation_labels, np.array(p_valid) > thresholds, beta=2, average='samples'), thresholds

    def fit(self, flow, epochs, lr, validation_data, validation_labels, resnet_layers_trainable=0, validation_steps=63, train_callbacks=[], batches=300):
        for layer in self.resnet.layers:
            layer.trainable = False
        for layer in self.resnet.layers[-resnet_layers_trainable:]:
            layer.trainable = False
        history = LossHistory()
        opt = Adam(lr=lr)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        self.model.fit_generator(flow, steps_per_epoch=batches, nb_epoch=epochs, callbacks=[history, earlyStopping] + train_callbacks,
            validation_data=validation_data, validation_steps=validation_steps)
        fb_score, thresholds = self.get_fbeta_score(validation_data, validation_labels, validation_steps, verbose=False)
        return [history.train_losses, history.val_losses, fb_score, thresholds]

    def save_weights(self, weight_file_path):
        self.model.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def predict_image(self, image):
        img = Image.fromarray(np.uint8(image))
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

    def optimise_f2_thresholds(self, y, p, verbose=True, resolution=100):
        """
        more details https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475

        :param y: labels for validation set
        :param p: prediction for validation set
        :param verbose:
        :param resolution:
        :return: thresholds array
        """
        print(type(p))
        print(type(y))

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

    def close(self):
        backend.clear_session()
