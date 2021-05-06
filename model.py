from tensorflow import keras
# from keras.layers import Reshape, Dense, Flatten, Dropout, ZeroPadding3D, Conv3D, MaxPool3D, BatchNormalization, Input,Convolution3D,Activation,GlobalAveragePooling3D
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential, load_model, Model
# from keras.optimizers import Adam, RMSprop , SGD
# from keras.layers.wrappers import TimeDistributed
# from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
# from keras.losses import categorical_crossentropy

class models():
    def __init__(self, nb_classes, model, input_shape):

        self.nb_classes = nb_classes
        # self.load_model = load_model

        if model == 'basic':
            print("Loading basic model...")
            self.input_shape = (28,28,1)
            self.model = self.basic()
        elif model == 'vgg16':
            print("Loading vgg16 model...")
            self. input_shape = (28,28,1)
            self.model = self.vgg16()

        self.model.summary()

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


    def basic(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=self.input_shape))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))

        return model

    def vgg16(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', input_shape=self.input_shape))
        model.add(keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        model.add(keras.layers.Dropout(rate=0.25))

        model.add(keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        model.add(keras.layers.Dropout(rate=0.25))

        model.add(keras.layers.Conv2D(256, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.Conv2D(256, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        model.add(keras.layers.Dropout(rate=0.25))

        model.add(keras.layers.Conv2D(512, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.Conv2D(512, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.Conv2D(512, (3,3), strides=(1,1), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        model.add(keras.layers.Dropout(rate=0.25))

        # model.add(keras.layers.Conv2D(512, (3,3), strides=(1,1), activation='relu'))
        # model.add(keras.layers.Conv2D(512, (3,3), strides=(1,1), activation='relu'))
        # model.add(keras.layers.Conv2D(512, (3,3), strides=(1,1), activation='relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        # model.add(keras.layers.Dropout(rate=0.25))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.25))
        model.add(keras.layers.Dense(self.nb_classes, activation='softmax'))

        return model