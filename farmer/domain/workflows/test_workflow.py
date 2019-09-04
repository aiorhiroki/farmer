from ..workflows.abstract_workflow import AbstractImageAnalyzer
from ..tasks.build_model_task import BuildModelTask
from ..tasks.set_train_env_task import SetTrainEnvTask
from ..tasks.read_annotation_task import ReadAnnotationTask
from ..tasks.eda_task import EdaTask
from ..tasks.predict_classification_task import PredictClassificationTask
from ..tasks.eval_classification_task import EvalClassificationTask
from ..tasks.eval_segmentation_task import EvalSegmentationTask
from ..model.task_model import Task


class TestWorkflow(AbstractImageAnalyzer):
    def __init__(self, config):
        super().__init__(config)

    def command(self):
        self.set_env()
        test_set = self.read_annotation()
        self.eda()
        model, base_model = self.build_model()
        result = self.model_execution(test_set, model, base_model)
        return self.output(result)

    def set_env(self):
        SetTrainEnvTask(self._config).command()

    def read_annotation(self):
        read_annotation = ReadAnnotationTask(self._config)
        test_set = read_annotation.command('test')
        return test_set

    def eda(self):
        EdaTask(self._config).command()

    def build_model(self):
        model, base_model = BuildModelTask(self._config).command()
        return model, base_model

    def model_execution(
        self, annotation_set, model, base_model, validation_set
    ):
        if self._config.task == Task.CLASSIFICATION:
            prediction = PredictClassificationTask(self._config).command(
                annotation_set, model
            )
            eval_report = EvalClassificationTask(self._config).command(
                prediction, annotation_set
            )
        elif self._config.task == Task.SEMANTIC_SEGMENTATION:
            eval_report = EvalSegmentationTask(self._config).command(
                annotation_set, model
            )
        return eval_report

    def output(self, result):
        return result
