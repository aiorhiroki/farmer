import shutil
from farmer import ncc
import os
import json
import numpy as np
import cv2
from tqdm import trange
from ..model.task_model import Task


class EdaTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set):
        self._do_compute_mean_std(annotation_set)
        self._do_save_params_task()
        self._do_post_config_task()

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
                if self.config.task != Task.OBJECT_DETECTION:
                    fw.write("class_name,class_id\n")
                for cls_id, class_name in enumerate(self.config.class_names):
                    fw.write(f"{class_name},{cls_id}\n")
        with open(f"{self.config.info_path}/mean_std.json", "w") as fw:
            json.dump(
                dict(
                    mean=list(self.config.mean),
                    std=list(self.config.std)
                ), fw, indent=2
            )

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

    def _do_compute_mean_std(self, annotation_set):
        """train set全体の平均と標準偏差をchannelごとに計算
        """
        train_set, _, _ = annotation_set
        if self.config.train_params.augmix:
            self.config.mean_std = True

        if self.config.input_data_type == 'image' and self.config.mean_std:
            if len(train_set) == 0:
                return
            elif len(self.config.mean) > 0 and len(self.config.std) > 0:
                return

            means = []
            pix_pow = np.zeros(3)
            for i in trange(len(train_set)):
                input_file, label = train_set[i]
                x = cv2.imread(input_file)
                x = cv2.resize(x, (self.config.width, self.config.height))
                x = x / 255.  # 正規化してからmean,stdを計算する
                means.append(np.mean(x, axis=(0, 1)))
                pix_pow += np.sum(np.power(x, 2), axis=(0, 1))
            pix_num = self.config.height * self.config.width * len(train_set)
            mean = np.mean(means, axis=(0))
            var_pix = (pix_pow / pix_num) - np.power(mean, 2)
            std = np.sqrt(var_pix)
            # convert BGR to RGB
            self.config.mean = mean[::-1]
            self.config.std = std[::-1]
