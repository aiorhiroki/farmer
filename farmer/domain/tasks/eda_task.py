import configparser
from farmer import ncc
import os
import dataclasses
import numpy as np
import cv2


class EdaTask:
    def __init__(self, config):
        self.config = config

    def command(self, train_set):
        self._do_save_params_task()
        self._do_post_config_task()
        self._do_compute_mean_std(train_set)

    def _do_save_params_task(self):
        parser = configparser.ConfigParser()
        config_dict = dataclasses.asdict(self.config)
        config_dict = {k: v for (k, v) in config_dict.items() if v}
        parser["project_settings"] = config_dict
        param_path = os.path.join(self.config.info_path, "parameter.txt")
        with open(param_path, mode="w") as configfile:
            parser.write(configfile)
        with open(f"{self.config.info_path}/classes.csv", "w") as fw:
            if self.config.train_colors:
                fw.write("class_name,class_id,color_id\n")
                for class_id, class_data in enumerate(self.config.train_colors):
                    if type(class_data) == int:
                        class_name = self.config.class_names[class_id]
                        color_id = class_data
                    elif type(class_data) == dict:
                        color_id, class_id = list(class_data.items())[0]
                        class_name = self.config.class_names[class_id]
                    fw.write(f"{class_name},{class_id},{color_id}\n")
            else:
                fw.write("class_name,class_id\n")
                for class_id, class_name in enumerate(self.config.class_names):
                    fw.write(f"{class_name},{class_id}\n")

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
        bgr_images = []
        for input_file, label in train_set:
            x = cv2.imread(input_file)
            x = cv2.resize(x, (self.config.width, self.config.height))
            x = x / 255.
            bgr_images.append(x)
        mean = np.mean(np.array(bgr_images), axis=(0, 1, 2))
        std = np.std(np.array(bgr_images), axis=(0, 1, 2))
        # convert BGR to RGB
        self.config.mean = mean[::-1]
        self.config.std = std[::-1]
