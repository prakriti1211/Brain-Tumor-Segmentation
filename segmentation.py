import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import cv2

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = "Brain/train"
TEST_PATH = "Brain/test"

TRAIN_IMAGES = os.listdir(TRAIN_PATH + '/image/')
n=len(TRAIN_IMAGES)

TRAIN_MASK = os.listdir(TRAIN_PATH + '/mask/')
n=len(TRAIN_MASK)


X_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


for i in tqdm(range(len(TRAIN_IMAGES))):
    img_name = TRAIN_IMAGES[i]
    img = imread(TRAIN_PATH + '/image/' + img_name)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant',
                 preserve_range=True)
    X_train[i] = img
    
for j in tqdm(range(len(TRAIN_MASK))):
    mask_name = TRAIN_MASK[j]
    mask = imread(TRAIN_PATH + '/mask/' + mask_name)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant',
                     preserve_range=True)
    Y_train[j] = mask


   
TEST_IMAGES = os.listdir(TEST_PATH + '/image/')
a=len(TEST_IMAGES)
X_test = np.zeros((a, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)


for i in tqdm(range(len(TEST_IMAGES))):
    img_name = TEST_IMAGES[i]
    img = imread(TEST_PATH + '/image/' + img_name)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant',
                 preserve_range=True)
    X_test[i] = img


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
x = tf.keras.layers.Lambda(lambda x : x / 255)(inputs)

# DownSampling
c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
p3 = MaxPooling2D((2,2))(c3)

c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
p4 = MaxPooling2D((2,2))(c4)

c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)


# UpSampling
u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=outputs)
model.compile(optimizer='adadelta', loss="binary_crossentropy",
              metrics=['accuracy'])
model.summary()


# Including CheckPoint
checkpoint = tf.keras.callbacks.ModelCheckpoint('model_checkpoints.h5',
                                                save_best_only=True,
                                                verbose=1)

# Including EarlyStopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]


results = model.fit(X_train, Y_train, validation_split=0.1, 
                    batch_size=16, epochs=30, callbacks=callbacks)


test_predictions = model.predict(X_test)


imshow(X_test[16])
plt.show()

imshow(test_predictions[16])
plt.show()

