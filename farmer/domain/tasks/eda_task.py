import shutil
from farmer import ncc
import os
import json
import numpy as np
import cv2
import random


class EdaTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set):
        train_set, _, _ = annotation_set
        self._do_save_params_task()
        self._do_post_config_task()
        if self.config.input_data_type == 'image' and self.config.mean_std:
            if len(train_set) > 2000:
                sample_train_set = random.sample(train_set, 2000)
            elif len(train_set) > 0:
                sample_train_set = train_set
            else:
                return
            self._do_compute_mean_std(sample_train_set)

    def _do_save_params_task(self):
        shutil.copy(self.config.config_path, self.config.info_path)
        with open(f"{self.config.info_path}/classes.csv", "w") as fw:
            if self.config.train_colors:
                fw.write("class_name,class_id,color_id\n")
                for cls_id, cls_data in enumerate(self.config.train_colors):
                    if type(cls_data) == int:
                        class_name = self.config.class_names[cls_id]
                        color_id = cls_data
                    elif type(cls_data) == dict:
                        color_id, class_id = list(cls_data.items())[0]
                        class_name = self.config.class_names[class_id]
                    fw.write(f"{class_name},{cls_id},{color_id}\n")
            else:
                fw.write("class_name,class_id\n")
                for cls_id, class_name in enumerate(self.config.class_names):
                    fw.write(f"{class_name},{cls_id}\n")
        with open(f"{self.config.info_path}/mean_std.json", "w") as fw:
            json.dump(dict(mean=self.config.mean, std=self.config.std), fw)

    def _do_post_config_task(self):
        # milk側にconfigを送る
        if self.config.train_id is None:
            return
        milk_client = ncc.utils.post_client.PostClient(
            root_url=self.config.milk_api_url
        )
        milk_client.post(
            params=dict(
                train_id=int(self.config.milk_id),
                nb_classes=self.config.nb_classes,
                height=self.config.height,
                width=self.config.width,
                result_path=os.path.abspath(self.config.result_path),
                class_names=self.config.class_names,
            ),
            route="first_config",
        )
        milk_client.close_session()

    def _do_compute_mean_std(self, train_set):
        """train set全体の平均と標準偏差をchannelごとに計算
        """
        if self.config.mean is not None and self.config.std is not None:
            return
        bgr_images = []
        for input_file, label in train_set:
            x = cv2.imread(input_file)
            x = cv2.resize(x, (self.config.width, self.config.height))
            x = x / 255.  # 正規化してからmean,stdを計算する
            bgr_images.append(x)
        mean = np.mean(bgr_images, axis=(0, 1, 2))
        std = np.std(bgr_images, axis=(0, 1, 2))
        # convert BGR to RGB
        self.config.mean = mean[::-1]
        self.config.std = std[::-1]
