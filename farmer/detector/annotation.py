import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET


def search_classes(xml_dir):
    classes = []
    xmls = [x for x in os.listdir(xml_dir) if not x == '.DS_Store']
    for xml in xmls:
        tree = ET.parse(os.path.join(xml_dir, xml))
        root = tree.getroot()
        for object_tree in root.findall('object'):
            class_name = object_tree.find('name').text
            classes.append(class_name)

    return sorted(list(set(classes)))


def make_classes(xml_dir):
    your_classes = search_classes(xml_dir)
    ssd_classes = [
        'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
        'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor'
    ]
    assert len(your_classes) < len(ssd_classes), 'Too many classes in your dataset.'
    ssd_classes[:len(your_classes)] = your_classes

    return ssd_classes


def to_onehot(classes, class_name):
    assert class_name in classes, 'unknown label: %s' % class_name
    one_hot_vector = [0] * len(classes)
    one_hot_vector[classes.index(class_name)] = 1

    return one_hot_vector


def to_dict(classes, xml_dir):
    targets = {}
    files = [x for x in os.listdir(xml_dir) if not x == '.DS_Store']
    for xml in files:
        bndboxes = []
        onehot_classes = []
        tree = ET.parse(os.path.join(xml_dir, xml))
        root = tree.getroot()
        image = root.find('filename').text
        size_tree = root.find('size')
        width = float(size_tree.find('width').text)
        height = float(size_tree.find('height').text)
        for object_tree in root.findall('object'):
            class_name = object_tree.find('name').text
            for bndbox in object_tree.iter('bndbox'):
                xmin = float(bndbox.find('xmin').text)/width
                ymin = float(bndbox.find('ymin').text)/height
                xmax = float(bndbox.find('xmax').text)/width
                ymax = float(bndbox.find('ymax').text)/height
            bndboxes.append([xmin, ymin, xmax, ymax])
            onehot_classes.append(to_onehot(classes, class_name))
        bndboxes = np.asarray(bndboxes)
        onehot_classes = np.asarray(onehot_classes)
        targets[image] = np.hstack((bndboxes, onehot_classes))

    return targets


def make_target(classes, xml_dir, output='./target.pkl'):
    target = to_dict(classes, xml_dir)
    pickle.dump(target, open(output, 'wb'))