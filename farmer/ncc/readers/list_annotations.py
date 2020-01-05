import os
from glob import glob
import csv

from .search_xml_profile import generate_target_csv


def classification_set(target_dir, data_list, class_names):
    IMAGE_EXTENTINS = ['.jpg', '.png']

    annotations = list()
    for class_id, class_name in enumerate(class_names):
        image_paths = list()
        for image_ex in IMAGE_EXTENTINS:
            for case_data in data_list:
                image_paths += glob(
                    os.path.join(
                        target_dir,
                        case_data,
                        class_name,
                        '*' + image_ex
                    )
                )
        class_annotation = [
            [image_path, class_id] for image_path in image_paths
        ]
        annotations.extend(class_annotation)

    return annotations


def segmentation_set(target_dir, data_list, image_dir, mask_dir):
    IMAGE_EXTENTINS = ['.jpg', '.png']
    annotations = list()
    mask_paths = list()

    for image_ex in IMAGE_EXTENTINS:
        for data_case in data_list:
            mask_paths += glob(
                os.path.join(
                    target_dir,
                    data_case,
                    mask_dir,
                    '*' + image_ex
                )
            )
    for mask_path in mask_paths:
        file_name, _ = os.path.splitext(os.path.basename(mask_path))
        mask_dir_path, _ = os.path.split(mask_path)
        parent_mask_dir_path, _ = os.path.split(mask_dir_path)
        image_dir_path = os.path.join(parent_mask_dir_path, image_dir)
        for image_ex in IMAGE_EXTENTINS:
            image_path = os.path.join(
                image_dir_path, file_name + image_ex
            )
            if os.path.exists(image_path):
                annotations.append([image_path, mask_path])

    return annotations


def detection_set(
    target_dir,
    data_list,
    image_dir,
    xml_dir,
    csv_file,
    class_names=None
):
    for data_case in data_list:
        xml_dir_path = os.path.join(target_dir, data_case, xml_dir)
        image_dir_path = os.path.join(target_dir, data_case, image_dir)
        generate_target_csv(
            xml_dir_path,
            image_dir_path,
            save_to=csv_file,
            class_names=class_names
        )


def data_set_from_annotation(annotation_file):
    """
    collect annotation files in target dir (target/data/class/image_file)
    :param annotation_file: annotation_csv_file
    :return: train_set: [image_file_path, label_idx]
             test_set: [image_file_path, label_idx]
    """
    data_set = list()
    with open(annotation_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data_set.append(row)
    return data_set
