import math
import cv2

# from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
from ..augmentation import segmentation_aug
from ..tasks import Task
from ..utils import ImageUtil

import torch.utils.data as data

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
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, input_image = video.read()

            if not ret:
                return None, None

            input_image = input_image / 255.0
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
                input_file, anti_alias=True
            )

        if self.task == Task.SEMANTIC_SEGMENTATION:
            label = self.image_util.read_image(
                label,
                normalization=False,
                train_colors=self.train_colors
            )

        if self.augmentation and len(self.augmentation) > 0:
            print("一旦、オーギュメンテーションなし")
            # input_image, label = segmentation_aug(
            #     input_image,
            #     label,
            #     self.input_shape,
            #     self.augmentation
            # )

        input_image_result = np.array(input_image, dtype=np.float32)
        label_result = self.image_util.cast_to_onehot(label)

        return input_image_result, label_result

    def __len__(self) -> int:
        return len(self.annotations)
