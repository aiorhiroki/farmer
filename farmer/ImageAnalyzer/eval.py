from keras.models import load_model
from utils.reader import train_test_files

import numpy as np
from PIL import Image

from ncc.metrics import show_matrix

_, test_set = train_test_files()
batch_size = 1
nb_categories = 3
size = (299, 299)


def generate_batch_arrays(annotations):

    while True:
        x, y = [], []
        for annotation in annotations:
            input_file, label = annotation
            input_image = read_image(input_file, anti_alias=True)  # 入力画像は高品質にリサイズ

            x.append(input_image)
            y.append(label)

            if len(x) == batch_size:
                yield process_input(x, y)
                x, y = [], []


def process_input(images_original, labels):
    # Cast to ndarray
    images_original = np.asarray(images_original, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    images_segmented = cast_to_onehot(labels)

    return images_original, images_segmented


def read_image(file_path, normalization=True, anti_alias=False):
    image = Image.open(file_path)
    # resize
    if size != image.size:
        image = image.resize(size, Image.ANTIALIAS) if anti_alias else image.resize(self.size)
    # delete alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = np.asarray(image)
    if normalization:
        image = image / 255.0

    return image


def cast_to_onehot(labels):
    if len(labels.shape) == 1:  # Classification
        one_hot = np.eye(nb_categories, dtype=np.uint8)
    else:  # Segmentation
        one_hot = np.identity(nb_categories, dtype=np.uint8)
    return one_hot[labels]


model = load_model('result/20190301_1340/model/last_model.h5')
prediction = model.predict_generator(generate_batch_arrays(test_set), steps=len(test_set)//batch_size)
prediction_cls = np.argmax(prediction, axis=1)
true_cls = []
for test_annotation in test_set:
    _, label_id = test_annotation
    true_cls.append(label_id)

show_matrix(true_cls, prediction_cls, ['normal', 'adenoma', 'cancer'], show_plot=False, save_file='confusion_matrix')
