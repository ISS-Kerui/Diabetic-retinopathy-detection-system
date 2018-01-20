# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from sklearn.svm import SVC
import sys
from keras.optimizers import Adam, SGD
import theano

import cv2
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# path to the model weights file.
weights_path = '../../h5/vgg16_weights.h5'
top_model_weights_path = '../../h5/eye_top.h5'

# dimensions of our images.
#self.img_width, self.img_height = 256, 256
# self.train_data_dir = '/Volumes/Echo/data/train_ds5_crop'
# self.validation_data_dir = '/Volumes/Echo/data/test_ds5_crop'
# self.nb_train_samples = 0
# nb_validation_samples = 0
# self.np_epoch


class TrainModel():

    def file_count(self, dirname, filter_types=[]):
        count = 0
        filter_is_on = False
        if filter_types != []:
            filter_is_on = True
        for item in os.listdir(dirname):
            abs_item = os.path.join(dirname, item)
            # print item
            if os.path.isdir(abs_item):
                # Iteration for dir
                count += self.file_count(abs_item, filter_types)
            elif os.path.isfile(abs_item):
                if filter_is_on:
                    # Get file's extension name
                    extname = os.path.splitext(abs_item)[1]
                    if extname in filter_types:
                        count += 1
                else:
                    count += 1
        return count

    def __init__(self, train_data_dir, validation_data_dir, optimizerType, learnRate, epoch_num, richText, button):
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.img_width = 256
        self.img_height = 256
        self.epoch_num = epoch_num
        self.optimizerType = optimizerType
        self.learnRate = learnRate
        self.richText = richText
        self.button = button

    def save_bottlebeck_features(self):
        self.nb_train_samples = self.file_count(self.train_data_dir)
        self.nb_validation_samples = self.file_count(self.validation_data_dir)
        datagen = ImageDataGenerator(
            rotation_range=360,
            rescale=1./255,
            # zoom_range=0.1,
        )
        datagen2 = ImageDataGenerator(

            rescale=1./255,

        )

       # sys.stdout = self.richText
        # build the VGG16 network
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(
            3, self.img_width, self.img_height)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call
        # model.load_weights(filename)
        assert os.path.exists(
            weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the
                # savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)]
                       for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)

        print('Model loaded.')

        train_generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
        print('generator ok.')
        bottleneck_features_train = model.predict_generator(
            train_generator, self.nb_train_samples)

        np.save(open('npy/bottleneck_features_train.npy', 'wb'),
                bottleneck_features_train)

        generator = datagen2.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
        bottleneck_features_validation = model.predict_generator(
            generator, self.nb_validation_samples)
        np.save(open('npy/bottleneck_features_validation.npy', 'wb'),
                bottleneck_features_validation)
        print('save_bottlebeck_features ok')
        # log.close()
        self.button.Enable()

    def train_top_model(self):

       # sys.stdout = self.richText
        train_data = np.load(open('npy/bottleneck_features_train.npy'))
        pic_num1 = []
        for label in os.listdir(self.train_data_dir):
            current_dir = os.path.join(self.train_data_dir, label)
            pic_num1.append(self.file_count(current_dir))

        train_labels = np.array(
            [0]*pic_num1[0] + [1]*pic_num1[1] + [2]*pic_num1[2]+[3]*pic_num1[3]+[4]*pic_num1[4])

        train_labels = np_utils.to_categorical(train_labels, 5)
        validation_data = np.load(
            open('npy/bottleneck_features_validation.npy'))
        pic_num2 = []
        for label in os.listdir(self.validation_data_dir):
            current_dir = os.path.join(self.validation_data_dir, label)
            pic_num2.append(self.file_count(current_dir))
        test_labels = np.array(
            [0] * pic_num2[0] + [1]*pic_num2[1]+[2]*pic_num2[2]+[3]*pic_num2[3]+[4]*pic_num2[4])
        test_labels = np_utils.to_categorical(test_labels, 5)

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='softmax'))
        if self.optimizerType == 'Adam':

            adam = Adam(lr=float(self.learnRate), beta_1=0.9,
                        beta_2=0.99, epsilon=1e-8)
            model.compile(loss='categorical_crossentropy',
                          optimizer=adam,
                          metrics=['accuracy'])
        elif self.optimizerType == 'SGD':
            sgd = SGD(lr=float(self.learnRate), beta_1=0.9,
                      beta_2=0.99, epsilon=1e-8)
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])
        model.fit(train_data, train_labels,
                  nb_epoch=int(self.epoch_num), batch_size=64,
                  validation_data=(validation_data, test_labels), verbose=2)
        model.save_weights('softmax.h5')
        print('train_top_model ok')
        self.button.Enable()

    def predict(self):
        self.predict_data_dir = 'predict/'
        nb_predict_samples = self.file_count(self.predict_data_dir)
        print nb_predict_samples
        datagen = ImageDataGenerator(rescale=1./255)

        # build the VGG16 network
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(
            3, self.img_width, self.img_height)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call
        # model.load_weights(filename)
        assert os.path.exists(
            weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the
                # savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)]
                       for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')
        generator = datagen.flow_from_directory(
            self.predict_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=32,
            class_mode=None,
            shuffle=True)
        print('generator ok.')

        bottleneck_features_predict = model.predict_generator(
            generator, nb_predict_samples)
        print('predict ok.')

        np.save(open('npy/bottleneck_features_predict.npy', 'wb'),
                bottleneck_features_predict)

        print('save_bottlebeck_features ok')
        self.predict_data = np.load(
            open('npy/bottleneck_features_predict.npy'))
        pic_num2 = []
        for label in os.listdir(self.predict_data_dir):
            current_dir = os.path.join(self.predict_data_dir, label)
            pic_num2.append(self.file_count(current_dir))
        predict_labels = np.array(
            [0] * pic_num2[0] + [1]*pic_num2[1]+[2]*pic_num2[2]+[3]*pic_num2[3]+[4]*pic_num2[4])

        #predict_labels = np_utils.to_categorical(predict_labels, 5)
        model_top = Sequential()
        model_top.add(Flatten(input_shape=self.predict_data.shape[1:]))
        model_top.add(Dense(128, activation='relu'))
        model_top.add(Dense(256, activation='relu'))
        model_top.add(Dropout(0.5))
        model_top.add(Dense(5, activation='softmax'))
        model_top.load_weights('softmax.h5')
        classes = model_top.predict_classes(self.predict_data)
        test_accuracy = np.mean(np.equal(predict_labels, classes))
        print("accuarcy:", test_accuracy)
        self.button.Enable()
