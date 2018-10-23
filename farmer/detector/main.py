import os
import cv2
import pickle
import argparse
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session

from models import SSD300
from utils import BBoxUtility
from losses import MultiboxLoss
from generator import SSDGenerator
from annotation import make_classes, make_target


# Search path
cwd = os.getcwd()
dirlist = [x for x in os.listdir(cwd) if os.path.isdir(x) == True]
for d in dirlist:
    if '.xml' in os.listdir(d)[0]:
        xml_dir = d
    if '.jpg' in os.listdir(d)[0]:
        img_dir = d

# Matplot config
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

# Make targets
classes = make_classes(xml_dir)
make_target(classes, xml_dir)

# Dataset
gt = pickle.load(open('target.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

# Parameter
base_lr = 3e-4
batch_size = 32
NUM_CLASSES = 21
input_shape = (300, 300, 3)

# Default boxes
priors = pickle.load(open('default_boxes.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# Train
def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'),
    keras.callbacks.LearningRateScheduler(schedule)
]

gen = SSDGenerator(
    gt, bbox_util, batch_size, img_dir,
    train_keys, val_keys, (input_shape[0], input_shape[1]), do_crop=True
)

model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('default_weights.hdf5', by_name=True)     

freeze = [
    'input_1', 'conv1_1', 'conv1_2', 'pool1',
    'conv2_1', 'conv2_2', 'pool2',
    'conv3_1', 'conv3_2', 'conv3_3', 'pool3'
]

for L in model.layers:
    if L.name in freeze:
        L.trainable = False
        
model.compile(
    optimizer=keras.optimizers.Adam(lr=base_lr),
    loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss
)

model.fit_generator(
    gen.generate(True),
    gen.train_batches//batch_size,
    epochs=100,
    verbose=1,
    callbacks=callbacks,
    validation_data=gen.generate(False),
    validation_steps=gen.val_batches//batch_size,
    workers=1
)

# Predict
test_set = sorted([x for x in os.listdir(img_dir) if not x == '.DS_Store'])
test_set = test_set[:5]
inputs, images = [], []

for i in range(len(test_set)):
    img = image.load_img(test_set[i], target_size=(300, 300))
    img = image.img_to_array(img)
    inputs.append(img.copy())
    images.append(imread(test_set[i]))

inputs = preprocess_input(np.array(inputs))
preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

for i, img in enumerate(images):
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.show()