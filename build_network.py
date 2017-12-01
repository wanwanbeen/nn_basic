import os
from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import metrics
from keras.regularizers import l2
K.set_image_dim_ordering('th')

os.environ['CUDA_VISIBLE_DEVICES']="0"

####################################################################
# Define network and model
####################################################################

def train_network(img_train, img_val, label_train, label_val, image_size, depth_level):
    model = get_network(image_size, depth_level)
    model_checkpoint = ModelCheckpoint('model.hdf5', monitor='loss', save_best_only=True)
    model.summary()
    model.fit(img_train, label_train, batch_size=8, epochs=10, verbose=1, shuffle=False,
              validation_data=[img_val,label_val])    
    return model

def get_network(img_size, depth_level):
    inputs = Input((1,img_size[0], img_size[1], img_size[2]))
    first_channel_num = 4
    conv1 = Convolution3D(first_channel_num, (3, 3, 3), activation='relu', padding = 'same',
                          kernel_regularizer=l2(0.00))(inputs)
    conv1 = Convolution3D(first_channel_num, (3, 3, 3), activation=None, padding = 'same',
                          kernel_regularizer=l2(0.00))(conv1)
    bn1 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                       beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(conv1)
    relu1 = Activation('relu')(bn1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu1)

    conv2 = Convolution3D(first_channel_num*2, (3, 3, 3), activation='relu', padding = 'same',
                          kernel_regularizer=l2(0.00))(pool1)
    conv2 = Convolution3D(first_channel_num*2, (3, 3, 3), activation=None, padding = 'same',
                          kernel_regularizer=l2(0.00))(conv2)
    bn2 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(conv2)
    relu2 = Activation('relu')(bn2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu2)

    conv3 = Convolution3D(first_channel_num*4, (3, 3, 3), activation='relu', padding = 'same',
                          kernel_regularizer=l2(0.00))(pool2)
    conv3 = Convolution3D(first_channel_num * 4, (3, 3, 3), activation=None, padding='same',
                          kernel_regularizer=l2(0.00))(conv3)
    bn3 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(conv3)
    relu3 = Activation('relu')(bn3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu3)

    conv4 = Convolution3D(first_channel_num*8, (3, 3, 3), activation='relu', padding = 'same',
                          kernel_regularizer=l2(0.00))(pool3)
    conv4 = Convolution3D(first_channel_num * 8, (3, 3, 3), activation=None, padding='same',
                          kernel_regularizer=l2(0.00))(conv4)
    bn4 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(conv4)
    relu4 = Activation('relu')(bn4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu4)

    conv5 = Convolution3D(first_channel_num * 16, (3, 3, 3), activation='relu', padding='same',
                          kernel_regularizer=l2(0.00))(pool4)
    conv5 = Convolution3D(first_channel_num * 16, (3, 3, 3), activation=None, padding='same',
                          kernel_regularizer=l2(0.00))(conv5)
    bn5 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros',
                             moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(conv5)
    relu5 = Activation('relu')(bn5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(relu5)
    
    if depth_level == 1:
        flatten1 = Flatten()(pool1)
    elif depth_level == 2:
        flatten1 = Flatten()(pool2)
    elif depth_level == 3:
        flatten1 = Flatten()(pool3)
    elif depth_level == 4:
        flatten1 = Flatten()(pool4)
    else:
        flatten1 = Flatten()(pool5)

    fc1 = Dense(1, activation='sigmoid', use_bias=True)(flatten1)

    model = Model(output=fc1,input=inputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])

    return model
