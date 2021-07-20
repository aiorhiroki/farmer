from mrcnn.config import Config
from mrcnn import model, utils

from imgaug import augmenters as iaa
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2


"""
docker exec -it maskrcnn_tf2 bash -c "cd $PWD && python farmer/domain/tasks/train_mrcnn.py \
--dataset=/mnt/cloudy_z/src/yalee/instrument_tip_detection_yalee/datasets/preprocessed/3instruments"
"""


class FarmerConfig(Config):
    NAME = "farmer"
    NUM_CLASSES = 1 + 3
    IMAGE_MAX_DIM = 640

class FarmerDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
        data_classes = ['linear', 'point', 'grasper']
        for class_id, data_class in enumerate(data_classes):
            self.add_class('farmer', class_id+1, data_class)

        # Add images
        images = sorted((Path(dataset_dir) / subset / "images").glob("*.png"))
        print("nb images", len(images))

        for image_path in images:
            mask_path = str(image_path).replace("/images/", "/labels/")
            if not Path(mask_path).exists():
                continue
            mask = cv2.imread(mask_path)
            if np.sum(mask) == 0:
                continue

            self.add_image(
                'farmer', 
                path=str(image_path), 
                image_id=str(image_path).rstrip(".jpg"),
                mask_path=mask_path,
                width=640, height=640
            )

    def load_image(self, image_id):
        image_info = self.image_info[image_id]
        width, height = image_info['width'], image_info['height']
        image = Image.open(str(image_info['path'])).resize((width, height))
        return np.array(image)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        mask = np.array(Image.open(str(image_info['mask_path'])))
        mask[mask == 191] = 1
        mask[mask == 192] = 2
        mask[mask == 193] = 3
        mask[mask == 194] = 4
        mask[mask == 195] = 5
        mask[mask == 196] = 6
        mask[mask > 6] = 0

        count = len(np.unique(mask)[1:])
        width, height = image_info['width'], image_info['height']
        out = np.zeros([height, width, count], dtype=np.uint8)

        mask_ids = list(np.unique(mask)[1:])
        for nb_mask, mask_id in enumerate(mask_ids):
            m_o = cv2.resize(np.array(mask == mask_id, dtype=np.uint8), (width, height))
            out[:, :, nb_mask] = m_o

        class_ids = np.unique(mask)[1:]
        class_ids[class_ids == 4] = 3
        class_ids[class_ids == 5] = 3
        class_ids[class_ids == 6] = 3

        return out.astype(np.int32), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]



parser = argparse.ArgumentParser(description='Train Mask R-CNN')
parser.add_argument('--dataset', required=True,
                    metavar="/path/to/coco/",
                    help='Directory of the train dataset')
args = parser.parse_args()

config = FarmerConfig()
config.display()
model = model.MaskRCNN(mode="training", config=config, model_dir="logs")

dataset_train = FarmerDataset()
dataset_train.load_dataset(args.dataset, "train")
dataset_train.prepare()

dataset_val = FarmerDataset()
dataset_val.load_dataset(args.dataset, "valid")
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

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=200,
            layers='all',
            augmentation=augmentation)
