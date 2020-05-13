from ..workflows.abstract_workflow import AbstractImageAnalyzer
from ..tasks.build_model_task import BuildModelTask
from ..tasks.set_train_env_task import SetTrainEnvTask
from ..tasks.read_annotation_task import ReadAnnotationTask
from ..tasks.eda_task import EdaTask
from ..tasks.train_task import TrainTask
from ..tasks.predict_classification_task import PredictClassificationTask
from ..tasks.predict_segmentation_task import PredictSegmentationTask
from ..tasks.evaluation_task import EvaluationTask
from ..tasks.output_result_task import OutputResultTask
from ..model.task_model import Task

import optuna
import numpy as np
import os
from keras.backend import clear_session


class TrainWorkflow(AbstractImageAnalyzer):
    def __init__(self, config):
        super().__init__(config)

    def command(self, trial=None):
        self.set_env_flow(trial)
        train_set, validation_set, test_set = self.read_annotation_flow()
        if trial is None or trial.number == 0:
            # 1回目のtrialのみEDAを行う
            self.eda_flow()
        model, base_model = self.build_model_flow(trial)
        result = self.model_execution_flow(
            train_set, model, base_model, validation_set, test_set, trial
        )
        return self.output_flow(result, trial)

    def set_env_flow(self, trial):
        SetTrainEnvTask(self._config).command(trial)
        print("set env flow done")

    def read_annotation_flow(self):
        read_annotation = ReadAnnotationTask(self._config)
        train_set = read_annotation.command("train")
        validation_set = read_annotation.command("validation")
        test_set = read_annotation.command("test")
        print("read annotation flow done")
        return train_set, validation_set, test_set

    def eda_flow(self):
        print("eda flow done")
        EdaTask(self._config).command()

    def build_model_flow(self, trial=None):
        if self._config.task == Task.OBJECT_DETECTION:
            # this flow is skipped for object detection at this moment
            # keras-retina command build model in model execution flow
            return None, None
        model, base_model = BuildModelTask(self._config).command(trial)
        print("build model flow done")
        return model, base_model

    def model_execution_flow(
        self,
        annotation_set, model, base_model, validation_set, test_set, trial
    ):
        if self._config.training:
            if self._config.task == Task.OBJECT_DETECTION:
                from keras_retinanet.bin import train
                annotations = f"{self._config.info_path}/train.csv"
                classes = f"{self._config.info_path}/classes.csv"
                val_annotations = f"{self._config.info_path}/validation.csv"
                train.main(
                    [
                        "--epochs", str(self._config.epochs),
                        "--steps", str(self._config.steps),
                        "--snapshot-path", self._config.model_path,
                        "csv", annotations, classes,
                        "--val-annotations", val_annotations
                    ]
                )
                trained_model = "{}/resnet50_csv_{:02d}.h5".format(
                    self._config.model_path, self._config.epochs
                )
            else:
                trained_model = TrainTask(self._config).command(
                    model, base_model,
                    annotation_set, validation_set,
                    trial
                )
        else:
            if self._config.task == Task.OBJECT_DETECTION:
                trained_model = self._config.trained_model_path
            else:
                trained_model = model
        if self._config.training and len(test_set) == 0:
            test_set = validation_set
        if len(test_set) == 0:
            return 0
        if self._config.task == Task.CLASSIFICATION:
            prediction = PredictClassificationTask(self._config).command(
                test_set, trained_model, self._config.save_pred
            )
            eval_report = EvaluationTask(self._config).command(
                test_set, prediction=prediction
            )
        elif self._config.task == Task.SEMANTIC_SEGMENTATION:
            PredictSegmentationTask(self._config).command(
                test_set, model=trained_model, trial=trial
            )
            eval_report = EvaluationTask(self._config).command(
                test_set, model=trained_model
            )
        elif self._config.task == Task.OBJECT_DETECTION:
            eval_report = EvaluationTask(self._config).command(
                test_set, model=trained_model
            )

        print("model execution flow done")
        print(eval_report)

        return eval_report

    def output_flow(self, result, trial):
        OutputResultTask(self._config).command(result, trial)
        print("output flow done")
        return result

    def optuna_command(self):
        # Logger for Optuna
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        # TODO: 任意のパスを設定したら保存されない問題．infoフォルダにいれたい．
        logger.addHandler(logging.FileHandler(
            "optuna_trials.log", mode="w")
        )
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.enable_default_handler()  # Stop showing logs in sys.stderr.

        study = optuna.create_study(direction='maximize')
        study.optimize(
            self.objective,
            n_trials=self._config.n_trials,
            timeout=self._config.timeout
        )
        print("Study statistics: ")
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        param_path = os.path.join(self._config.info_path, "best_hyperparams.txt")

        with open(param_path, mode="w") as configfile:
            print(f"  Value: {trial.value}")
            configfile.write(f"optuna objective value = {trial.value}\n")
            print("  Params: ")
            configfile.write(f"optuna set params = {trial.params}\n")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        return study

    def objective(self, trial):
        clear_session()
        result = self.command(trial)
        if self._config.task == Task.CLASSIFICATION:
            return result["accuracy"]
        elif self._config.task == Task.SEMANTIC_SEGMENTATION:
            return np.mean(result["dice"][1:])
        else:
            raise NotImplementedError
