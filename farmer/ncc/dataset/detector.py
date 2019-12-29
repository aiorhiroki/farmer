import csv
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
from six import raise_from
from skimage.io import imread
import xml.etree.ElementTree as ET


def search_class_names(xml_dir):
    """ Search class names from xml files.
    """
    class_names = []
    if not os.path.basename(xml_dir):
        xml_dir = os.path.join(xml_dir, '*.xml')
    for xml in glob(xml_dir):
        tree = ET.parse(xml)
        root = tree.getroot()
        for object_tree in root.findall('object'):
            class_name = object_tree.find('name').text
            class_names.append(class_name)

    return sorted(list(set(class_names)))


def make_class_csv(xml_dir=None, class_names=None, save_path='./class.csv'):
    """ Make class csv.
    """
    if xml_dir and not class_names:
        class_names = search_class_names(xml_dir)
    with open(save_path, 'w') as csv:
        for class_name in class_names:
            csv.write(class_name + ',' + str(class_names.index(class_name)) + '\n')


def make_target_csv(xml_dir, img_dir, save_path='./target.csv'):
    """ Make target csv.
    """
    if not os.path.basename(xml_dir):
        xml_dir = os.path.join(xml_dir, '*.xml')
    if os.path.basename(img_dir):
        img_dir = os.path.dirname(os.path.abspath(img_dir))
    with open(save_path, 'w') as csv:
        for xml in sorted(glob(xml_dir)):
            tree = ET.parse(xml)
            root = tree.getroot()
            image = root.find('filename').text
            image = os.path.basename(image)
            image_path = os.path.join(img_dir, image)
            if not os.path.exists(image_path):
                print(image_path + 'does not exists.')
                continue
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                for bndbox in object_tree.iter('bndbox'):
                    xmin = str(bndbox.find('xmin').text)
                    ymin = str(bndbox.find('ymin').text)
                    xmax = str(bndbox.find('xmax').text)
                    ymax = str(bndbox.find('ymax').text)
                target = [image_path, xmin, ymin, xmax, ymax, class_name]
                csv.write(','.join([str(t) for t in target]) + '\n')


def clip_box(xml_dir):
    """ Clip bounding box coords into range of the image size.
    """
    if not os.path.basename(xml_dir):
        xml_dir = os.path.join(xml_dir, '*.xml')
    for xml in glob(xml_dir):
        tree = ET.parse(xml)
        root = tree.getroot()
        size_tree = root.find('size')
        width = size_tree.find('width').text
        height = size_tree.find('height').text
        for object_tree in root.findall('object'):
            for bndbox in object_tree.iter('bndbox'):
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                if xmin < 0:
                    bndbox.find('xmin').text = 0
                    tree.write(xml)
                    print('xmin < 0, clip ' + str(xml))
                if ymin < 0:
                    bndbox.find('ymin').text = 0
                    tree.write(xml)
                    print('ymin < 0, clip ' + str(xml))
                if xmax > float(width):
                    bndbox.find('xmax').text = width
                    tree.write(xml)
                    print('xman > width, clip ' + str(xml))
                if ymax > float(height):
                    bndbox.find('ymax').text = height
                    tree.write(xml)
                    print('ymax > height, clip ' + str(xml))


def read_annotations(csv_path):
    """ Read annotations from the csv_reader.
    """
    with open(csv_path, 'r') as file:
        gt = {}
        for line, row in enumerate(csv.reader(file, delimiter=',')):
            line += 1
            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)
            if img_file not in gt:
                gt[img_file] = []
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue
            gt[img_file].append({'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'class': class_name})
    return gt


def display_images(csv_path, imgs_path=None, class_name=None, class_count=0, show_bbox=True, num=1):
    """ Display the given set of images.
    imgs_path: list of image pathes.
    class_name: class you want to display.
    class_count: number of appearance of class to an image.
    """
    gt = read_annotations(csv_path)
    if not imgs_path:
        imgs_path = list(gt.keys())
    if not type(imgs_path) is list:
        imgs_path = [imgs_path]
        num = len(imgs_path)
    random.shuffle(imgs_path)

    if class_name:
        new_imgs_path = []
        count = 0
        for img_path in imgs_path:
            class_names = []
            values = gt[img_path]
            for value in values:
                class_names.append(value['class'])
            if class_names.count(class_name) > class_count:
                new_imgs_path.append(img_path)
                count += 1
            if count == num:
                break
        imgs_path = new_imgs_path

    for idx, img_path in enumerate(imgs_path[:num], 1):
        print('No.'+str(idx)+': '+img_path)
        plt.figure(figsize=(12, 12))
        plt.imshow(imread(img_path))
        ax = plt.gca()
        if show_bbox:
            for data in gt[img_path]:
                x1, x2, y1, y2, class_name = data['x1'], data['x2'], data['y1'], data['y2'], data['class']
                ax.text(x1, y1, class_name, bbox={'facecolor':'red', 'alpha':0.5})
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
        plt.show()