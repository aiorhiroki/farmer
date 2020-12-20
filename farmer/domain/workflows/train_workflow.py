from ..workflows.abstract_workflow import AbstractImageAnalyzer
from ..tasks.build_model_task import BuildModelTask
from ..tasks.set_train_env_task import SetTrainEnvTask
from ..tasks.read_annotation_task import ReadAnnotationTask
from ..tasks.eda_task import EdaTask
from ..tasks.train_task import TrainTask
from ..tasks.predict_classification_task import PredictClassificationTask
from ..tasks.predict_segmentation_task import PredictSegmentationTask
from ..tasks.predict_detection_task import PredictDetectionTask
from ..tasks.output_result_task import OutputResultTask
from ..model.task_model import Task
import tensorflow as tf


class TrainWorkflow(AbstractImageAnalyzer):
    def __init__(self, config, trial=None):
        super().__init__(config)

    def command(self, trial=None):
        self.set_env_flow(trial)
        annotation_set = self.read_annotation_flow()
        self.eda_flow(annotation_set)
        model = self.build_model_flow()
        result = self.model_execution_flow(annotation_set, model, trial)
        return self.output_flow(result)

    def set_env_flow(self, trial):
        print("SET ENV FLOW ... ", end="")
        self._config = SetTrainEnvTask(self._config).command(trial)
        print("DONE")

    def read_annotation_flow(self):
        print("READ ANNOTATION FLOW ... ")
        read_annotation = ReadAnnotationTask(self._config)
        train_set = read_annotation.command("train")
        validation_set = read_annotation.command("validation")
        test_set = read_annotation.command("test")
        print("DONE")
        return train_set, validation_set, test_set

    def eda_flow(self, annotation_set):
        print("EDA FLOW ... ", end="")
        EdaTask(self._config).command(annotation_set)
        print("DONE")
        print("MEAN:", self._config.mean, "- STD: ", self._config.std)

    def build_model_flow(self):
        print("BUILD MODEL FLOW ... ")
        if self._config.task == Task.OBJECT_DETECTION:
            # this flow is skipped for object detection at this moment
            # keras-retina command build model in model execution flow
            return None, None
        if self._config.multi_gpu:
            with tf.distribute.MirroredStrategy().scope():
                model = BuildModelTask(self._config).command()
        else:
            model = BuildModelTask(self._config).command()
        print("DONE\n")
        return model

    def model_execution_flow(self, annotation_set, model, trial):
        train_set, val_set, test_set = annotation_set
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
                    model, train_set, val_set, trial)
        else:
            if self._config.task == Task.OBJECT_DETECTION:
                trained_model = self._config.trained_model_path
            else:
                trained_model = model
        if self._config.training and len(test_set) == 0:
            test_set = val_set
        if len(test_set) == 0:
            return 0
        if self._config.task == Task.CLASSIFICATION:
            eval_report = PredictClassificationTask(self._config).command(
                test_set, trained_model, self._config.save_pred
            )
        elif self._config.task == Task.SEMANTIC_SEGMENTATION:
            eval_report = PredictSegmentationTask(self._config).command(
                test_set, model=trained_model
            )
        elif self._config.task == Task.OBJECT_DETECTION:
            eval_report = PredictDetectionTask(self._config).command(
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
