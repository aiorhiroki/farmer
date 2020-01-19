from abc import ABC, abstractmethod


class AbstractImageAnalyzer(ABC):
    @abstractmethod
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def command(self):
        pass

    @abstractmethod
    def set_env_flow(self):
        pass

    @abstractmethod
    def read_annotation_flow(self):
        pass

    @abstractmethod
    def eda_flow(self):
        pass

    @abstractmethod
    def build_model_flow(self):
        pass

    @abstractmethod
    def model_execution_flow(
        self, annotation_set, model, base_model, validation_set=None
    ):
        pass

    @abstractmethod
    def output_flow(self, result):
        # 返り値のフォーマット
        pass
