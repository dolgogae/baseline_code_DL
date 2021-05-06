import tensorflow as tf
from tensorflow import keras
import numpy as np
# from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from model import models

img_size = 28
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# X_train_full = np.reshape(X_train_full, (-1,img_size,img_size,1))
# y_train_full = np.reshape(y_train_full, (-1,img_size,img_size,1))

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# print(y_train.shape)

X_valid = X_valid.reshape(-1,img_size,img_size,1)
X_train = X_train.reshape(-1,img_size,img_size,1)
# y_valid = y_valid.reshape(-1,img_size,img_size,1)
# y_train = y_train.reshape(-1,img_size,img_size,1)

def train(model, nb_classes, image_shape=(28,28,1), batch_size=32, nb_epoch=100):

    checkpointer = keras.callbacks.ModelCheckpoint(filepath='checkpoints/weight.hdf5', monitor='val_loss', verbose=1,
                                    save_best_only=True)

    early_stopper = keras.callbacks.EarlyStopping(patience=100)

    train_model = models(nb_classes, model, image_shape)
    train_model.model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=[early_stopper,checkpointer],
                        validation_data=(X_valid, y_valid))

def main():
    nb_classes = 10

    model = 'vgg16'

    # print(X_valid.shape)

    train(model, nb_classes)

if __name__ == '__main__':
    main()
