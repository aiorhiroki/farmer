import configparser
import ncc
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
        parser['project_settings'] = dataclasses.asdict(self.config)
        with open('parameter.txt', mode='w') as configfile:
            parser.write(configfile)

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
                class_names=self.config.class_names
            ),
            route='first_config'
        )
        milk_client.close_session()
