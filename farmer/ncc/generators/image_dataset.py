import cv2

import numpy as np
from ..augmentation import segmentation_aug_pytorch
from ..tasks import Task
from ..utils import ImageUtil

import torch.utils.data as data
from torchvision import transforms

# PyTorch版では、モデルの方に設定するパラメータが含まれている。　
# Kerasと混同してしまう？ので不要パラメータ削除予定
# -> LSTM対応やる際、1つのミニバッチで複数読み込む必要がでてくるa
# -> batch_sizeを削除する予定だったが、撤回して残す。


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
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # frame_id番目のframeを表示

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
                input_file, 
                normalization=False,
                anti_alias=True
            )

        if self.task == Task.SEMANTIC_SEGMENTATION:
            label = self.image_util.read_image(
                label,
                normalization=False,
                train_colors=self.train_colors
            )

        if self.augmentation and len(self.augmentation) > 0:
            input_image, label = segmentation_aug_pytorch(
                input_image,
                label,
                self.input_shape,
                self.augmentation
            )

        labels_onehot = self.image_util.cast_to_onehot(label.squeeze())
        input_image_tensor = transforms.functional.to_tensor(input_image)
        label_tensor = transforms.functional.to_tensor(labels_onehot)

        return input_image_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.annotations)
