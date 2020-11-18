import os
import yaml


class OutputResultTask:
    def __init__(self, config):
        self.config = config

    def command(self, result, trial=None):
        self._do_write_result_task(result, trial)

    def _do_write_result_task(self, result, trial):
        self.config.result = result

        if self.config.optuna:
            param_path = os.path.join(self.config.trial_result_path, "parameter.yaml")
            with open(param_path, mode="w") as configfile:
                yaml.dump(self.config, configfile)
                configfile.write(
                    f"\n optuna trial#{self.config.trial_number}\n")
                configfile.write(
                    f"optuna set params = {self.config.trial_params}\n")

        else:
            param_path = os.path.join(self.config.info_path, "parameter.yaml")
            with open(param_path, mode="w") as configfile:
                yaml.dump(self.config, configfile)
