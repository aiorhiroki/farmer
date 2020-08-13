import os
import yaml


class OutputResultTask:
    def __init__(self, config):
        self.config = config

    def command(self, result, trial=None):
        self._do_write_result_task(result, trial)

    def _do_write_result_task(self, result, trial):
        param_path = os.path.join(self.config.info_path, "parameter.yaml")
        self.config.result = result
        with open(param_path, mode="w") as configfile:
            yaml.dump(self.config, configfile)
            if self.config.trial_number is not None and \
                        self.config.trial_params is not None:
                configfile.write(f"\noptuna trial#{self.config.trial_number}\n")
                configfile.write(f"optuna set params = {self.config.trial_params}\n")
