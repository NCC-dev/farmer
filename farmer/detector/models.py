"""Keras implementation of SSD."""

import keras
import keras.backend as K
from keras.layers import Activation
from keras.layers import AtrousConv2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

from layers import Normalize, PriorBox


def SSD300(input_shape, num_classes=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    net = {}

    # Block 1
    input_tensor = input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor # (None, 300, 300, 3)
    net['conv1_1'] = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(net['input']) # (None, 300, 300, 64)
    net['conv1_2'] = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(net['conv1_1']) # (None, 300, 300, 64)
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(net['conv1_2']) # (None, 150, 150, 64)

    # Block 2
    net['conv2_1'] = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(net['pool1']) # (None, 150, 150, 128)
    net['conv2_2'] = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(net['conv2_1']) # (None, 150, 150, 128)
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(net['conv2_2']) # (None, 75, 75, 128)

    # Block 3
    net['conv3_1'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(net['pool2']) # (None, 75, 75, 256)
    net['conv3_2'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(net['conv3_1']) # (None, 75, 75, 256)
    net['conv3_3'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(net['conv3_2']) # (None, 75, 75, 256)
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(net['conv3_3']) # (None, 38, 38, 256)

    # Block 4
    net['conv4_1'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(net['pool3']) # (None, 38, 38, 512)
    net['conv4_2'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(net['conv4_1']) # (None, 38, 38, 512)
    net['conv4_3'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(net['conv4_2']) # (None, 38, 38, 512)
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(net['conv4_3']) # (None, 19, 19, 512)

    # Block 5
    net['conv5_1'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(net['pool4']) # (None, 19, 19, 512)
    net['conv5_2'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(net['conv5_1']) # (None, 19, 19, 512)
    net['conv5_3'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(net['conv5_2']) # (None, 19, 19, 512)
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='pool5')(net['conv5_3']) # (None, 19, 19, 512)

    # FC6
    net['fc6'] = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(net['pool5']) # (None, 19, 19, 1024)
    # x = Dropout(0.5, name='drop6')(x)

    # FC7
    net['fc7'] = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(net['fc6']) # (None, 19, 19, 1024)
    # x = Dropout(0.5, name='drop7')(x)

    # Block 6
    net['conv6_1'] = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(net['fc7']) # (None, 19, 19, 256)
    net['conv6_2'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_2', strides=(2, 2))(net['conv6_1']) # (None, 10, 10, 512)

    # Block 7
    net['conv7_1'] = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(net['conv6_2']) # (None, 10, 10, 128)
    net['conv7_2'] = ZeroPadding2D()(net['conv7_1']) # (None, 12, 12, 128)
    net['conv7_2'] = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv7_2', strides=(2, 2))(net['conv7_2']) # (None, 5, 5, 256)

    # Block 8
    net['conv8_1'] = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(net['conv7_2']) # (None, 5, 5, 128) 
    net['conv8_2'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv8_2', strides=(2, 2))(net['conv8_1']) # (None, 3, 3, 256)

    # Last Pool
    net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2']) # (None, 256)


    # Prediction 1 from conv4_3
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3']) # mbox_loc, mbox_conf, priorboxへ入力される, (None, 38, 38, 512)
    # mbox_loc
    num_priors = 3
    net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(net['conv4_3_norm']) # (None, 38, 38, 12)
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    # mbox_conf
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(net['conv4_3_norm']) # (None, 38, 38, 63)
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    # priorbox
    net['conv4_3_norm_mbox_priorbox'] = PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')(net['conv4_3_norm'])


    # Prediction 2 from fc7
    # mbox_loc
    num_priors = 6
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, (3, 3), padding='same', name='fc7_mbox_loc')(net['fc7']) # (None, 19, 19, 24)
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    # mbox_conf
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(net['fc7']) # (None, 19, 19, 126)
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    # priorbox
    net['fc7_mbox_priorbox'] = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')(net['fc7'])

    # Prediction 3 from conv6_2
    # mbox_loc
    num_priors = 6
    net['conv6_2_mbox_loc'] = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(net['conv6_2']) # (None, 10, 10, 24)
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # mbox_conf
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['conv6_2_mbox_conf'] = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(net['conv6_2']) # (None, 10, 10, 126)
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])
    # priorbox
    net['conv6_2_mbox_priorbox'] = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')(net['conv6_2'])


    # Prediction 4 from conv7_2
    # mbox_loc
    num_priors = 6
    net['conv7_2_mbox_loc'] = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(net['conv7_2']) # (None, 5, 5, 24)
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # mbox_conf
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['conv7_2_mbox_conf'] = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(net['conv7_2']) # (None, 5, 5, 126)
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    # priorbox
    net['conv7_2_mbox_priorbox'] = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')(net['conv7_2'])


    # Prediction 5 from conv8_2
    # mbox_loc
    num_priors = 6
    net['conv8_2_mbox_loc'] = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(net['conv8_2']) # (None, 3, 3, 24)
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    # mbox_conf
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['conv8_2_mbox_conf'] = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=name)(net['conv8_2']) # (None, 3, 3, 126)
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])
    # priorbox
    net['conv8_2_mbox_priorbox'] = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')(net['conv8_2'])


    # Prediction 6 from pool6
    # mbox_loc
    num_priors = 6
    net['pool6_mbox_loc_flat'] = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    # mbox_conf
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['pool6_mbox_conf_flat'] = Dense(num_priors * num_classes, name=name)(net['pool6'])
    # priorbox
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    net['pool6_reshaped'] = Reshape(target_shape, name='pool6_reshaped')(net['pool6']) # (None, 1, 1, 256)
    net['pool6_mbox_priorbox'] = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')(net['pool6_reshaped'])


    # Gather all predictions
    net['mbox_loc'] = keras.layers.concatenate([net['conv4_3_norm_mbox_loc_flat'], # (None, 17328) = 4 * 4332
                             net['fc7_mbox_loc_flat'], # (None, 8664) = 4 * 2166
                             net['conv6_2_mbox_loc_flat'], # (None, 2400) = 4 * 600
                             net['conv7_2_mbox_loc_flat'], # (None, 600) = 4 * 150
                             net['conv8_2_mbox_loc_flat'], # (None, 216) = 4 * 54
                             net['pool6_mbox_loc_flat']], # (None, 24) = 4 * 6
                             axis=1, name='mbox_loc') # (None, 29232) = 4 * 7308

    net['mbox_conf'] = keras.layers.concatenate([net['conv4_3_norm_mbox_conf_flat'], # (None, 90972) = 21 * 4332
                              net['fc7_mbox_conf_flat'], # (None, 45486) = 21 * 2166
                              net['conv6_2_mbox_conf_flat'], # (None, 12600) = 21 * 600
                              net['conv7_2_mbox_conf_flat'], # (None, 3150) = 21 * 150
                              net['conv8_2_mbox_conf_flat'], # (None, 1134) = 21 * 54
                              net['pool6_mbox_conf_flat']], # (None, 126) = 21 * 6
                              axis=1, name='mbox_conf') # (None, 153468) = 21 * 7308

    net['mbox_priorbox'] = keras.layers.concatenate([net['conv4_3_norm_mbox_priorbox'], # (None, 4332, 8)
                                  net['fc7_mbox_priorbox'], # (None, 2166, 8)
                                  net['conv6_2_mbox_priorbox'], # (None, 600, 8)
                                  net['conv7_2_mbox_priorbox'], # (None, 150, 8)
                                  net['conv8_2_mbox_priorbox'], # (None, 54, 8)
                                  net['pool6_mbox_priorbox']], # (None, 6, 8)
                                  axis=1, name='mbox_priorbox') # (None, 7308, 8)
    
    # num_boxesを定義
    if hasattr(net['mbox_loc'], '_keras_shape'): # 属性を持つかどうか
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4 # ._keras_shape = (None, 29232)
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    
    # predictions
    net['mbox_loc'] = Reshape((num_boxes, 4), name='mbox_loc_final')(net['mbox_loc']) # 回帰, (None, 7308, 4)
    net['mbox_conf'] = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(net['mbox_conf']) # 分類なのでsoftmax, (None, 7308, 21)
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = keras.layers.concatenate([net['mbox_loc'], net['mbox_conf'], net['mbox_priorbox']], axis=2, name='predictions') # (None, 7308, 33)
    
    # return model
    model = Model(net['input'], net['predictions'])
    return model
