from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mmdet.core.evaluation import eval_map
from mrcnn import visualize

from imgaug import augmenters as iaa
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

from farmer.ncc.readers import segmentation_set
from farmer.ncc.generators import MaskrcnnDataset

"""
# command
nohup docker exec -t maskrcnn_tf2 bash -c "cd $PWD && env CUDA_VISIBLE_DEVICES=0 \
python farmer/domain/tasks/train_mrcnn.py train" > train0.out &
"""

with open("example/yamls/mrcnn-instruments.yaml") as yamlfile:
    yaml_config = yaml.safe_load(yamlfile)


if yaml_config["TRAIN_DIRS"] is None or yaml_config["VAL_DIRS"] is None:
    dataset_dirs = os.listdir(yaml_config["DATASET"])
    split_line = int(0.8*len(dataset_dirs))
    yaml_config["TRAIN_DIRS"] = dataset_dirs[:split_line]
    yaml_config["VAL_DIRS"] = dataset_dirs[split_line:]

class TrainConfig(Config):
    NAME = "farmer"
    NUM_CLASSES = 1 + max(yaml_config["CLASS_IDS"].values())
    IMAGE_MAX_DIM = 640
    USE_MINI_MASK = False


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def evaluate_det(model, dataset, config, eval_type="bbox", image_ids=None):
    RESULTS_DIR = "results/forceps"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    nb_classes = config.NUM_CLASSES - 1
    print("class names: ", dataset.class_names)

    image_ids = image_ids or dataset.image_ids
    annotations = list()
    det_results = list()
    for i, image_id in tqdm(enumerate(image_ids)):
        # Load image
        image, _, gt_class_id, gt_bbox, mask = modellib.load_image_gt(
            dataset_val, config, image_id)
        gt_class_id = gt_class_id - 1
        annotations.append(dict(bboxes=gt_bbox, labels=gt_class_id))
        # Run detection
        r = model.detect([image], verbose=0)[0]
        r['class_ids'] = r['class_ids'] - 1
        det_result = [list() for _ in range(nb_classes)]
        num_det = r['rois'].shape[0]
        for det_id in range(num_det):
            class_id = r['class_ids'][det_id]
            for det_cls in range(nb_classes):
                if class_id == det_cls:
                    det_result[det_cls].append(
                        list(r['rois'][det_id]) + [float(r['scores'][det_id])])
        for det_cls in range(nb_classes):
            if len(det_result[det_cls]) == 0:
                det_result[det_cls] = np.empty((0, 5))
            else:
                det_result[det_cls] = np.array(det_result[det_cls])
        det_results.append(det_result)

        # to check prediction
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")


        """
        # to check groud truth
        visualize.display_instances(
            image, gt_bbox, mask, gt_class_id+1,
            dataset.class_names, [1.0 for _ in range(len(gt_bbox))],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(
          submit_dir, os.path.basename(dataset.image_info[image_id]["id"])))
        """

    mean_ap, _ = eval_map(det_results, annotations, iou_thr=0.7)
    print(mean_ap)


if yaml_config["TRAINING"]:
    config = TrainConfig()
    mode = "training"
else:
    config = InferenceConfig()
    mode = "inference"
config.display()

model = modellib.MaskRCNN(mode=mode, config=config, model_dir="logs")
if yaml_config["TRAINING"]:
    model_path = model.get_imagenet_weights()
else:
    model_path = yaml_config["TRAINED_MODEL"]
model.load_weights(model_path, by_name=True)

train_annos = segmentation_set(yaml_config["DATASET"], yaml_config["TRAIN_DIRS"], yaml_config["IMAGE_DIR"], yaml_config["MASK_DIR"])
dataset_train = MaskrcnnDataset(train_annos, yaml_config["CLASS_IDS"], yaml_config["CLASS_NAMES"])
dataset_train.load_dataset()
dataset_train.prepare()

train_annos = segmentation_set(yaml_config["DATASET"], yaml_config["VAL_DIRS"], yaml_config["IMAGE_DIR"], yaml_config["MASK_DIR"])
dataset_val = MaskrcnnDataset(train_annos, yaml_config["CLASS_IDS"], yaml_config["CLASS_NAMES"])
dataset_val.load_dataset()
dataset_val.prepare()

augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])

if yaml_config["TRAINING"]:
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=160,
                augmentation=augmentation,
                layers='all')

evaluate_det(model, dataset_val, config)
