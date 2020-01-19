from glob import glob
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image


def search_from_dir(target_dir):
    files = []
    for image_suffix in ['jpg', 'png', 'jpeg']:
        # target_dir/train(test)/class_name/*.jpg
        image_paths = target_dir + '/*/*.' + image_suffix
        files += glob(image_paths)
    return search_image_profile(files)


def search_from_annotation(annotation_file):
    annotation_df = pd.read_csv(annotation_file).values.T
    files = annotation_df[0]
    class_index = annotation_df[1]

    class_names = np.unique(class_index)

    return search_image_profile(files), class_names


def search_image_profile(files):
    if len(files) > 10000:
        files = files[:10000]  # ignore large image files

    height_list, width_list, channel_list = [], [], []
    for file in files:
        image = Image.open(file)
        width, height = image.size[:2]
        channel = 1 if len(np.array(image).shape) == 2 else 3
        height_list.append(height)
        width_list.append(width)
        channel_list.append(channel)

    height_median = int(np.median(height_list))
    width_median = int(np.median(width_list))
    counter = Counter(channel_list)
    channel_most = int(counter.most_common(1)[0][0])

    return height_median, width_median, channel_most


def search_image_colors(files, min_count=1):
    # 画像の色を取得します。min_countで少ない色をフィルターします。
    if len(files) > 10000:
        files = files[:10000]  # ignore large image files
    color_list = list()
    for image_file in files:
        image = Image.open(image_file)
        colors = image.getcolors()
        # color is None if using over 255 colors
        if colors is None:
            continue
        for count, color in colors:
            if count >= min_count and color not in color_list:
                color_list.append(color)
    return sorted(color_list)
