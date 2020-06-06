import shutil
from farmer import ncc
import os
import dataclasses


class EdaTask:
    def __init__(self, config):
        self.config = config

    def command(self):
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
                fw.write("class_name,class_id\n")
                for class_id, class_name in enumerate(self.config.class_names):
                    fw.write(f"{class_name},{cls_id}\n")

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
