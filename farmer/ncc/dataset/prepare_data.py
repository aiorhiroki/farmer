import os
import csv
import cv2
import numpy as np

from keras.datasets import *


def prepare_data(module, nb_image=100, annotation_file=True):

    if module == 'mnist':
        names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    elif module == 'cifar10':
        names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    else:
        raise ValueError('We don\'t prepare that dataset yet.')

    _find_and_save_image(x_train, y_train, nb_image, names, module+'/train', annotation_file=annotation_file)
    _find_and_save_image(x_test, y_test, nb_image, names, module+'/test', annotation_file=annotation_file)

    return module


def _find_and_save_image(images, labels, nb_image, names, phase_folder, annotation_file=True):

    annotation = [['file_path', 'class_index']]
    for class_index, class_name in enumerate(names):
        print('find and save ', class_name)

        # save image in '.png' file
        save_dir = os.path.join(phase_folder, class_name)
        os.makedirs(save_dir, exist_ok=True)

        cls_image = images[np.where(labels.ravel() == class_index)]
        np.random.shuffle(cls_image)

        for i, image in enumerate(cls_image[:nb_image]):
            file_path = os.path.join(save_dir, str(i) + '.png')
            abs_path = os.path.abspath(file_path)
            cv2.imwrite(abs_path, image)
            annotation.append([abs_path, class_index])

    if annotation_file:
        with open(phase_folder+'_annotation.csv', 'w') as fw:
            print(phase_folder, 'annotation file saved in dataset/annotation.csv')
            writer = csv.writer(fw)
            writer.writerows(annotation)
