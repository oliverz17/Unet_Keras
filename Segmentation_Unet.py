#Oliver Z.
#Zurich, Spring 2018

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib
from utils import dice_coef_loss, dice_coef
matplotlib.use('Agg')

K.clear_session()
K.set_image_data_format('channels_last')

#set parameters
img_rows = 480
img_cols = 480
img_channels = 3
epochs = 150
batch_size = 4


#build model
class Unet():
    def __init__(self, img_rows, img_cols, img_channels, epochs, batch_size):
        self.img_height = img_rows
        self.img_width = img_cols
        self.img_channels = img_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.create_model()
        self.model = self.model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])


        self.data_path_images_train = r'/images_train/'
        self.data_path_maps_train = r'/maps_train/'
        self.data_path_images_val = r'/images_val/'
        self.data_path_maps_val = r'/maps_val/'

        self.images_train = [self.data_path_images_train + f for f in os.listdir(self.data_path_images_train) if f.endswith('.png')]
        self.number_images_train = len(self.images_train)
        self.number_batches_train = math.ceil(self.number_images_train/self.batch_size) #math.ceil does round to the next higher integer, if not already a integer

        self.images_val = [self.data_path_images_val + f for f in os.listdir(self.data_path_images_val) if f.endswith('.png')]
        self.number_images_val = len(self.images_val)
        self.number_batches_val = math.ceil(self.number_images_val/self.batch_size)

    def create_model(self):
        inputs = Input((self.img_height, self.img_width, self.img_channels))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        return model


    def preprocess_images(self, paths):

        image_batch = np.zeros((len(paths), self.img_height, self.img_width, self.img_channels))

        for i, path in enumerate(paths):
            #if '.DS_Store' in path:
            #    continue
            original_image = cv2.imread(path, 1)
            img = cv2.resize(original_image, (self.img_height, self.img_width))
            img = np.array(([img]), dtype='float32')
            image_batch[i] = img

        return image_batch

    def preprocess_maps(self, paths):

        mask_batch = np.zeros((len(paths), self.img_height, self.img_width))

        for i, path in enumerate(paths):
            #if '.DS_Store' in path:
            #    continue
            original_mask = cv2.imread(path,0)
            mask = cv2.resize(original_mask, (self.img_height, self.img_width))
            mask = np.array(([mask]), dtype='float32')
            mask /= 255
            mask_batch[i] = mask


        mask_batch = np.expand_dims(mask_batch, axis=4)
        return mask_batch

    def generator_train(self, batch_size):

        images = [self.data_path_images_train + f for f in os.listdir(self.data_path_images_train) if f.endswith('.png')]
        maps = [self.data_path_maps_train + f for f in os.listdir(self.data_path_maps_train) if f.endswith('.png')]

        images.sort()
        maps.sort()


        counter = 0
        while True:
            yield self.preprocess_images(images[counter:counter + batch_size]), self.preprocess_maps(maps[counter:counter + batch_size])
            counter = (counter + batch_size) % len(images)


    def generator_val(self, batch_size):

        images = [self.data_path_images_val + f for f in os.listdir(self.data_path_images_val) if f.endswith('.png')]
        maps = [self.data_path_maps_val + f for f in os.listdir(self.data_path_maps_val) if f.endswith('.png')]

        images.sort()
        maps.sort()

        counter = 0
        while True:
            yield self.preprocess_images(images[counter:counter + batch_size]), self.preprocess_maps(maps[counter:counter + batch_size])
            counter = (counter + batch_size) % len(images)


    def train(self):
        model_checkpoint = ModelCheckpoint('weights_checkpoint.h5', monitor='val_loss', save_best_only=True)
        train_history = self.model.fit_generator(self.generator_train(batch_size=self.batch_size), steps_per_epoch = self.number_batches_train, epochs=self.epochs, validation_data=self.generator_val(batch_size=self.batch_size), validation_steps = self.number_batches_val, callbacks=[model_checkpoint])
        self.model.save('model.h5')
        self.model.save_weights('weights.h5')


        with open('History_train.pickle', 'wb') as history_file:
            pickle.dump(train_history.history, history_file)

            
if __name__ == '__main__':

    #Build Unet with given parameters:
    init_model = Unet(img_rows, img_cols, img_channels, epochs, batch_size)
    #Train model
    init_model.train()
