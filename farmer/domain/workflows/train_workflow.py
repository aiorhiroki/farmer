from ..domain.workflows.abstract_workflow import AbstractImageAnalyzer
from ..domain.tasks.build_model_task import BuildModelTask
from ..domain.tasks.set_train_env_task import SetTrainEnvTask
from ..domain.tasks.read_annotation_task import ReadAnnotationTask
from ..domain.tasks.eda_task import EdaTask
from ..domain.tasks.train_task import TrainTask
from ..tasks.predict_classification_task import PredictClassificationTask
from ..tasks.eval_claddification_task import EvalClassificationTask
from ..tasks.eval_segmentation_task import EvalSegmentationTask
from ..model.task_model import Task


class TrainWorkflow(AbstractImageAnalyzer):
    def __init__(self, config):
        super().__init__(config)

    def command(self):
        self.set_env()
        train_set, validation_set = self.read_annotation()
        self.eda()
        model, base_model = self.build_model()
        result = self.model_execution(
            train_set, model, base_model, validation_set
        )
        return self.output(result)

    def set_env(self):
        SetTrainEnvTask(self._config).command()

    def read_annotation(self):
        read_annotation = ReadAnnotationTask(self._config)
        train_set = read_annotation.command('train')
        validation_set = read_annotation.command('validation')

        return train_set, validation_set

    def eda(self):
        EdaTask(self._config).command()

    def build_model(self):
        model, base_model = BuildModelTask(self._config).command()
        return model, base_model

    def model_execution(
        self, annotation_set, model, base_model, validation_set
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

    def output(self, result):
        return result
