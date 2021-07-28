from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mmdet.core.evaluation import eval_map

from imgaug import augmenters as iaa
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
import os
import datetime
from tqdm import tqdm

from farmer.ncc.readers import segmentation_set
from farmer.ncc.generators import MaskrcnnDataset

"""
# for train
docker exec -it maskrcnn_tf2 bash -c "cd $PWD && python farmer/domain/tasks/train_mrcnn.py train \
--dataset=/mnt/cloudy_z/src/yalee/instrument_tip_detection_yalee/datasets/preprocessed/3instruments"

# for evaluate
docker exec -it maskrcnn_tf2 bash -c "cd $PWD && python farmer/domain/tasks/train_mrcnn.py eval \
--dataset=/mnt/cloudy_z/src/yalee/instrument_tip_detection_yalee/datasets/preprocessed/3instruments"
"""

CLASS_IDS = {191:1, 192:2, 193:3, 194:3, 195:3, 196:3}


parser = argparse.ArgumentParser(description='Train Mask R-CNN')
parser.add_argument("command",
                    metavar="<command>",
                    help="'train' or 'evaluate' on MS COCO")
parser.add_argument('--dataset', required=True,
                    metavar="/path/to/datset/",
                    help='Directory of the train dataset')
args = parser.parse_args()


class TrainConfig(Config):
    NAME = "farmer"
    NUM_CLASSES = 1 + max(CLASS_IDS.values())
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

    print("nb classes: ", config.NUM_CLASSES)

    image_ids = image_ids or dataset.image_ids
    annotations = list()
    det_results = list()
    for i, image_id in tqdm(enumerate(image_ids)):
        # Load image
        image, _, gt_class_id, gt_bbox, _ = modellib.load_image_gt(
            dataset_val, config, image_id)
        gt_class_id = gt_class_id - 1
        annotations.append(dict(bboxes=gt_bbox, labels=gt_class_id))
        # Run detection
        r = model.detect([image], verbose=0)[0]
        r['class_ids'] = r['class_ids'] - 1
        cls1_det = list()
        cls2_det = list()
        cls3_det = list()
        num_det = r['rois'].shape[0]
        for det_id in range(num_det):
            class_id = r['class_ids'][det_id]
            if class_id == 0:
                cls1_det.append(list(r['rois'][det_id]) + [float(r['scores'][det_id])])
            elif class_id == 1:
                cls2_det.append(list(r['rois'][det_id]) + [float(r['scores'][det_id])])
            elif class_id == 2:
                cls3_det.append(list(r['rois'][det_id]) + [float(r['scores'][det_id])])
        if len(cls1_det) == 0:
            cls1_det = np.empty((0, 5))
        else:
            cls1_det = np.array(cls1_det)
        if len(cls2_det) == 0:
            cls2_det = np.empty((0, 5))
        else:
            cls2_det = np.array(cls2_det)
        if len(cls3_det) == 0:
            cls3_det = np.empty((0, 5))
        else:
            cls3_det = np.array(cls3_det)
        det_results.append([cls1_det, cls2_det, cls3_det])

        """
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(
          submit_dir, os.path.basename(dataset.image_info[image_id]["id"])))
        """
    mean_ap, _ = eval_map(det_results, annotations, iou_thr=0.7)
    print(mean_ap)


if args.command == "train":
    config = TrainConfig()
    mode = "training"
else:
    config = InferenceConfig()
    mode = "inference"


config.display()
model = modellib.MaskRCNN(mode=mode, config=config, model_dir="logs")
model_path = model.get_imagenet_weights()
model.load_weights(model_path, by_name=True)

train_annos = segmentation_set(args.dataset, ["train"], "images", "labels")
dataset_train = MaskrcnnDataset(train_annos, CLASS_IDS)
dataset_train.load_dataset()
dataset_train.prepare()

train_annos = segmentation_set(args.dataset, ["valid"], "images", "labels")
dataset_val = MaskrcnnDataset(train_annos, CLASS_IDS)
dataset_val.load_dataset()
dataset_val.prepare()

augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])

if args.command == "train":
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
else:
    evaluate_det(model, dataset_val, config)
