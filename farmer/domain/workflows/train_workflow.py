import os

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


class TrainWorkflow(AbstractImageAnalyzer):
    def __init__(self, config, trial):
        super().__init__(config)
        if self._config.op_backbone is not None:
            self._config.backbone = trial.suggest_categorical(
                'backbone', self._config.op_backbone
            )

        if self._config.op_learning_rate is not None:
            # logスケールで変化
            if len(self._config.op_learning_rate) == 2:
                self._config.learning_rate = trial.suggest_loguniform(
                    'learning_rate', *self._config.op_learning_rate
                )
            # 線形スケールで変化
            elif len(self._config.op_learning_rate) == 3:
                self._config.learning_rate = trial.suggest_discrete_uniform(
                    'learning_rate', *self._config.op_learning_rate
                )

        if self._config.op_optimizer is not None:
            self._config.optimizer = trial.suggest_categorical(
                'optimizer', self._config.op_optimizer
            )

        if self._config.op_loss is not None:
            self._config.loss = trial.suggest_categorical(
                'loss', self._config.op_loss
            )

        if self._config.op_batch_size is not None:
            self._config.batch_size = int(trial.suggest_discrete_uniform(
                'batch_size', *self._config.op_batch_size)
            )

        if trial:
            # init
            self._config.model_path = os.path.join(self._config.result_path, self._config.model_dir)
            self._config.learning_path = os.path.join(self._config.result_path, self._config.learning_dir)
            self._config.image_path = os.path.join(self._config.result_path, self._config.image_dir)

            # result_dir/trial#/learning/
            self._config.learning_path = self._config.learning_path.replace("/learning", f"/trial{trial.number}/learning")
            # result_dir/trial#/model/
            self._config.model_path = self._config.model_path.replace("/model", f"/trial{trial.number}/model")
            # result_dir/trial#/image/
            self._config.image_path = self._config.image_path.replace("/image", f"/trial{trial.number}/image")

            self._config.trial_number = trial.number
            self._config.trial_params = trial.params

        

    def command(self, trial=None):
        self.set_env_flow()
        train_set, validation_set, test_set = self.read_annotation_flow()
        self.eda_flow(train_set)
        model, base_model = self.build_model_flow()
        result = self.model_execution_flow(
            train_set, model, base_model, validation_set, test_set, trial
        )
        return self.output_flow(result)

    def set_env_flow(self):
        print("SET ENV FLOW ... ", end="")
        SetTrainEnvTask(self._config).command()
        print("DONE")

    def read_annotation_flow(self):
        print("READ ANNOTATION FLOW ... ")
        read_annotation = ReadAnnotationTask(self._config)
        train_set = read_annotation.command("train")
        validation_set = read_annotation.command("validation")
        test_set = read_annotation.command("test")
        print("DONE")
        return train_set, validation_set, test_set

    def eda_flow(self, train_set):
        print("EDA FLOW ... ", end="")
        EdaTask(self._config).command(train_set)
        print("DONE")
        print("MEAN:", self._config.mean, "- STD: ", self._config.std)

    def build_model_flow(self):
        print("BUILD MODEL FLOW ... ")
        if self._config.task == Task.OBJECT_DETECTION:
            # this flow is skipped for object detection at this moment
            # keras-retina command build model in model execution flow
            return None, None
        model, base_model = BuildModelTask(self._config).command()
        print("DONE\n")
        return model, base_model

    def model_execution_flow(
        self,
        annotation_set, model, base_model, validation_set, test_set, trial
    ):
        print("MODEL EXECUTION FLOW ... ")
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
                test_set, model=trained_model
            )
            eval_report = EvaluationTask(self._config).command(
                test_set, model=trained_model
            )
        elif self._config.task == Task.OBJECT_DETECTION:
            eval_report = EvaluationTask(self._config).command(
                test_set, model=trained_model
            )
        print("DONE")
        print(eval_report)

        return eval_report

    def output_flow(self, result):
        print("OUTPUT FLOW ... ", end="")
        OutputResultTask(self._config).command(result)
        print("DONE")
        return result
