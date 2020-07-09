import os
import yaml


class OutputResultTask:
    def __init__(self, config):
        self.config = config

    def command(self, result):
        self._do_write_result_task(result)

    def _do_write_result_task(self, result):
        param_path = os.path.join(self.config.info_path, "parameter.yaml")
        self.config.result = result
        with open(param_path, mode="w") as configfile:
            yaml.dump(self.config, configfile)
