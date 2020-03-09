import os
os.environ["CUDA_VISIBLE_DEVICES"]=""



import tensorflow as tf
from model import model
from glob import glob
import numpy as np
from config import touching_dir,not_touching_dir

n_pos=len(glob(not_touching_dir+'/*'))
n_neg=len(glob(touching_dir+'/*'))

print(n_pos,n_neg)

total=float(n_pos+n_neg)/2.
weights={0:total/n_neg,
         1:total/n_pos}

img_gen=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                          width_shift_range=0.03,
                                                          height_shift_range=0.03,
                                                          brightness_range=None,
                                                          shear_range=1,
                                                          zoom_range=0.05,
                                                          channel_shift_range=0.0,
                                                          fill_mode='nearest',
                                                          cval=0.0,
                                                          horizontal_flip=True,
                                                          preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
                                                          data_format='channels_last',
                                                          validation_split=0.3,
                                                          dtype='float32')

batch_size=32
train_gen=img_gen.flow_from_directory('labeled_data/',target_size=(96,96),class_mode='binary',classes=['not_touching','touching'], subset='training',batch_size=batch_size)
valid_gen=img_gen.flow_from_directory('labeled_data/',target_size=(96,96),class_mode='binary',classes=['not_touching','touching'], subset='validation',batch_size=batch_size)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)


model.fit_generator(train_gen,epochs=1000,validation_data=valid_gen,callbacks=[es],class_weight=weights)

for layers in model.layers:
    layers.trainable=True

adam=tf.keras.optimizers.Adam(lr=1e-5)
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

model.fit_generator(train_gen,epochs=1000,validation_data=valid_gen,callbacks=[es],class_weight=weights)

model.save_weights('model_train_v2.h5')
