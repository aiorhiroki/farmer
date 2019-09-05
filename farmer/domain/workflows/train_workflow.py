from ..workflows.abstract_workflow import AbstractImageAnalyzer
from ..tasks.build_model_task import BuildModelTask
from ..tasks.set_train_env_task import SetTrainEnvTask
from ..tasks.read_annotation_task import ReadAnnotationTask
from ..tasks.eda_task import EdaTask
from ..tasks.train_task import TrainTask
from ..tasks.predict_classification_task import PredictClassificationTask
from ..tasks.eval_classification_task import EvalClassificationTask
from ..tasks.eval_segmentation_task import EvalSegmentationTask
from ..model.task_model import Task


class TrainWorkflow(AbstractImageAnalyzer):
    def __init__(self, config):
        super().__init__(config)

    def command(self):
        self.set_env_flow()
        train_set, validation_set = self.read_annotation_flow()
        self.eda()
        model, base_model = self.build_model_flow()
        result = self.model_execution_flow(
            train_set, model, base_model, validation_set
        )
        return self.output_flow(result)

    def set_env_flow(self):
        SetTrainEnvTask(self._config).command()

    def read_annotation_flow(self):
        read_annotation = ReadAnnotationTask(self._config)
        train_set = read_annotation.command('train')
        validation_set = read_annotation.command('validation')

        return train_set, validation_set

    def eda_flow(self):
        EdaTask(self._config).command()

    def build_model_flow(self):
        model, base_model = BuildModelTask(self._config).command()
        return model, base_model

    def model_execution_flow(
        self,
        annotation_set,
        model,
        base_model,
        validation_set
    ):
        trained_model = TrainTask(self._config).command(
            model, base_model, annotation_set, validation_set
        )

        if self._config.task == Task.CLASSIFICATION:
            prediction = PredictClassificationTask(self._config).command(
                annotation_set, trained_model
            )
            eval_report = EvalClassificationTask(self._config).command(
                prediction, annotation_set
            )
        elif self._config.task == Task.SEMANTIC_SEGMENTATION:
            eval_report = EvalSegmentationTask(self._config).command(
                annotation_set, trained_model
            )

        return eval_report

    def output_flow(
        self,
        result
    ):
        return result
