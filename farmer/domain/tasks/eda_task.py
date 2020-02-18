import configparser
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
        parser = configparser.ConfigParser()
        config_dict = dataclasses.asdict(self.config)
        config_dict = {k: v for (k, v) in config_dict.items() if v}
        parser["project_settings"] = config_dict
        param_path = os.path.join(self.config.info_path, "parameter.txt")
        with open(param_path, mode="w") as configfile:
            parser.write(configfile)
        with open(f"{self.config.info_path}/classes.csv", "w") as fw:
            for class_id, class_name in enumerate(self.config.class_names):
                if self.config.train_colors:
                    class_data = self.config.train_colors[class_id]
                    if type(class_data) == int:
                        class_id = class_data
                    elif type(class_data) == dict:
                        class_id, _ = list(class_data.items())[0]
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
