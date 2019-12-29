"""
input image: values index each class.
image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = array([[0, 0, 0, 205],
                                                         [205, 0, 0, 220],
                                                         [0, 220, 220, 0]])
return image :
image = array([[
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 1, 0]
            ]])
"""

import numpy as np


def segmentation_input(image):
    image = image[:, :, 0]
    class_values = np.unique(image)[1:]  # 0 is ignored
    output_image = np.zeros(image.shape + (len(class_values),))
    for i, class_value in enumerate(class_values):
        output_image[:, :, i] = np.where(image == class_value, 1, 0)

    return output_image
