from ..workflows.abstract_workflow import AbstractImageAnalyzer
from ..tasks.build_model_task import BuildModelTask
from ..tasks.set_train_env_task import SetTrainEnvTask
from ..tasks.read_annotation_task import ReadAnnotationTask
from ..tasks.eda_task import EdaTask
from ..tasks.predict_classification_task import PredictClassificationTask
from ..tasks.predict_segmentation_task import PredictSegmentationTask
from ..model.task_model import Task


class PredictWorkflow(AbstractImageAnalyzer):
    def __init__(self, config):
        super().__init__(config)

    def command(self):
        self.set_env_flow()
        test_set = self.read_annotation_flow()
        self.eda_flow()
        model, base_model = self.build_model_flow()
        result = self.model_execution_flow(test_set, model, base_model)
        return self.output_flow(result)

    def set_env_flow(self):
        SetTrainEnvTask(self._config).command()

    def read_annotation_flow(self):
        read_annotation = ReadAnnotationTask(self._config)
        test_set = read_annotation.command('test')
        return test_set

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
        if self._config.task == Task.CLASSIFICATION:
            prediction = PredictClassificationTask(self._config).command(
                annotation_set, model
            )
        elif self._config.task == Task.SEMANTIC_SEGMENTATION:
            prediction = PredictSegmentationTask(self._cofing).command(
                annotation_set, model
            )
        return prediction

    def output_flow(
        self,
        result
    ):
        return result
