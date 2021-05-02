import yaml


class OutputResultTask:
    def __init__(self, config):
        self.config = config

    def command(self, result, trial=None):
        self._do_write_result_task(result, trial)

    def _do_write_result_task(self, result, trial):
        self.config.result = result
        param_path = self.config.info_path
        if self.config.optuna:
            trial_number = self.config.trial_number
            param_path = f"{self.config.result_path}/trial{trial_number}"

        with open(f"{param_path}/parameter.yaml", mode="w") as configfile:
            yaml.dump(self.config, configfile)
            if self.config.optuna:
                configfile.write(f"\n optuna trial#{trial_number}")
                configfile.write(
                    f"\n optuna params = {self.config.trial_params}")
