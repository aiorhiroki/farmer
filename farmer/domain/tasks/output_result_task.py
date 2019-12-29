import os


class OutputResultTask:
    def __init__(self, config):
        self.config = config

    def command(self, result):
        self._do_write_result_task(result)

    def _do_write_result_task(self, result):
        param_path = os.path.join(self.config.info_path, "parameter.txt")
        with open(param_path, mode="a") as configfile:
            configfile.write(f"result = {result}")
