import os


class OutputResultTask:
    def __init__(self, config):
        self.config = config

    def command(self, result, trial=None):
        self._do_write_result_task(result, trial)

    def _do_write_result_task(self, result, trial):
        param_path = os.path.join(self.config.info_path, "parameter.txt")
        with open(param_path, mode="a") as configfile:
            if trial:
                configfile.write(f"\noptuna trial#{trial.number}\n")
                configfile.write(f"optuna set params = {trial.params}\n")

            configfile.write(f"nb_train = {self.config.nb_train_data}\n")
            configfile.write(
                f"nb_validation = {self.config.nb_validation_data}\n")
            configfile.write(f"nb_test = {self.config.nb_test_data}\n")
            configfile.write(f"result = {result}\n")
