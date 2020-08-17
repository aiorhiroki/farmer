import os
import cv2
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


def classification_video_set(
    target_dir,
    data_list,
    csv_file,
    class_names,
    skip_frame=30,
    time_format="datetime"
):
    target_dir = os.path.abspath(target_dir)
    annotations = list()
    for data_case in data_list:
        data_case_path = f"{target_dir}/{data_case}"
        if csv_file:
            csv_path = f"{data_case_path}/{csv_file}"
        else:
            csv_paths = glob(f"{data_case_path}/*.csv")
            if len(csv_paths) == 0:
                continue
            else:
                csv_path = csv_paths[0]
        video_path = None
        data_case_files = os.listdir(f"{data_case_path}")
        for data_case_file in data_case_files:
            if data_case_file.endswith((".mp4", ".avi")):
                video_path = f"{data_case_path}/{data_case_file}"
        if video_path is None:
            continue
        fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
        with open(csv_path, "r") as fr:
            reader = csv.reader(fr)
            next(reader)
            for start_time, end_time, class_name in reader:
                if class_name not in class_names:
                    continue
                class_id = class_names.index(class_name)
                if time_format == "datetime":
                    start_time = _str_time_to_frame(start_time, fps)
                    end_time = _str_time_to_frame(end_time, fps)
                else:
                    start_time, end_time = int(start_time), int(end_time)
                annotations.extend(
                    [
                        [video_path, frame, class_id]
                        for frame in range(start_time, end_time)
                        if skip_frame == 0 or frame % skip_frame == 0
                    ]
                )
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
            if 'label' in file_name:
                file_name = file_name.replace('label', 'movieFrame')
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
    class_names
):
    target_dir = os.path.abspath(target_dir)
    for data_case in data_list:
        xml_dir_path = os.path.join(target_dir, data_case, xml_dir)
        image_dir_path = os.path.join(target_dir, data_case, image_dir)
        generate_target_csv(
            xml_dir_path,
            image_dir_path,
            save_to=csv_file,
            class_names=class_names
        )


def _str_time_to_frame(str_time, fps):
    hour, minute, second = str_time.split(":")
    frame_id = int(hour)*60*60*fps + int(minute)*60*fps + int(second)*fps
    return int(frame_id)


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
