import cv2

from ..augmentation import segmentation_aug_pytorch
from ..tasks import Task
from ..utils import ImageUtil

import torch
import torch.utils.data as data
from torchvision import transforms


class ImageDataset(data.Dataset):
    def __init__(
        self,
        annotations: list,
        input_shape: (int, int),
        nb_classes: int,
        task: str,
        batch_size: int,
        augmentation=list(),
        train_colors=list(),
        input_data_type="image"
    ) -> None:

        self.annotations = annotations
        self.batch_size = batch_size    # PyTorchでは使用しない
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.task = task
        self.augmentation = augmentation
        self.train_colors = train_colors
        self.input_data_type = input_data_type

    def __getitem__(self, idx: int):
        input_file, label = self.annotations[idx]

        # input_file is [image_path] or [video_path, frame_id]
        # label is mask_image_path or class_id
        if self.input_data_type == "video":
            video_path, frame_id = input_file
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, input_image = video.read()

            if not ret:
                return None, None

            # (with,height) for cv2.resize
            resize_shape = self.input_shape[::-1]

            if input_image.shape[:2] != resize_shape:
                input_image = cv2.resize(
                    input_image,
                    resize_shape,
                    interpolation=cv2.INTER_LANCZOS4
                )
        else:
            input_image = self.image_util.read_image(
                input_file, normalization=False, anti_alias=True
            )

        input_image = transforms.functional.to_tensor(input_image)

        if self.task == Task.SEMANTIC_SEGMENTATION:
            label = self.image_util.read_image(
                label,
                normalization=False,
                train_colors=self.train_colors
            )

        label_raw = torch.from_numpy(label)
        label_tmp = torch.from_numpy(
            self.image_util.cast_to_onehot(label)
        )

        # [height, width, channel] -> [channel, height, width]
        label_tmp = label_tmp.permute(2, 0, 1)

        if self.augmentation and len(self.augmentation) > 0:
            input_image_augmented, label_augmented = segmentation_aug_pytorch(
                input_image,
                label_tmp,
                self.input_shape,
                self.augmentation
            )
        else:
            input_image_augmented = input_image
            label_augmented = label_tmp

        return input_image_augmented, label_augmented, label_raw

    def __len__(self) -> int:
        return len(self.annotations)
