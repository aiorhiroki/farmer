import os
import glob
import yaml
from farmer.ncc.tasks import Task
from farmer.ncc.mlflow_wrapper.mlflow_client_wrapper import MlflowClientWrapper

class OutputResultTask:
    def __init__(self, config):
        self.config = config

    def command(self, result, trial=None):
        self._do_write_result_task(result, trial)

        if self.config.mlflow:
            if not MlflowClientWrapper.is_running():
                MlflowClientWrapper.create_run(experiment_name=self.config.experiment_name,
                                                run_name=self.config.run_name,
                                                user_name=self.config.user_name)
            self._do_log_mlflow_task()
            MlflowClientWrapper.end_run()

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

    def _do_log_mlflow_task(self):
        print('[I] OutputResultTask._do_log_mlflow_task')
        
        MlflowClientWrapper.log_params({
            "model_name": self.config.train_params.model_name,
            "backbone": self.config.train_params.backbone,
            "activation": self.config.train_params.activation,
            "optimizer": self.config.train_params.optimizer,
            "class_names": self.config.class_names,
            "epochs": self.config.epochs,
            "loss": self.config.train_params.loss.get('functions'),
            "batch_size": self.config.train_params.batch_size,
            "learning_rate": self.config.train_params.learning_rate,
            "nb_train_data": self.config.nb_train_data,
            "nb_validation_data": self.config.nb_validation_data,
            "nb_test_data": self.config.nb_test_data,
        })

        MlflowClientWrapper.set_tags({
            "mlflow.note.content": self.config.description,
            "version": self.config.version
        })
        
        if self.config.optuna:
            MlflowClientWrapper.set_tags({
                "optuna trial": f"trial#{self.config.trial_number}"
            })
        
        MlflowClientWrapper.save_artifacts_to_mlruns(self.config.info_path, artifact_dir_name="info")
        MlflowClientWrapper.save_artifacts_to_mlruns(self.config.learning_path, artifact_dir_name="learning")
        MlflowClientWrapper.save_artifacts_to_mlruns(self.config.model_path, artifact_dir_name="model")
        
        dice_result_paths = glob.glob(os.path.join(self.config.image_path, 'test/*dice*'))
        if dice_result_paths:
            for p in dice_result_paths:
                MlflowClientWrapper.save_artifact_to_mlruns(p, artifact_dir_name="test")
        
        if self.config.data_dvc_path:
            for data_name, dvc_path in self.config.data_dvc_path.items():
                if (data_name is not None) & (dvc_path is not None):
                    MlflowClientWrapper.save_artifact_to_mlruns(dvc_path, artifact_dir_name=f"dvc/data/{data_name}")
        
        if self.config.task == Task.SEMANTIC_SEGMENTATION:
            MlflowClientWrapper.log_metrics_with_array(self.config.result)
        
        print('[O] OutputResultTask._do_log_mlflow_task')