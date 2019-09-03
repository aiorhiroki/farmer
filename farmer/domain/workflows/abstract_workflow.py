from abc import ABC, abstractmethod


class AbstractImageAnalyzer(ABC):
    @abstractmethod
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def command(self):
        pass

    @abstractmethod
    def set_env(self):
        pass

    @abstractmethod
    def read_annotation(self):
        pass

    @abstractmethod
    def eda(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def model_execution(
        self, annotation_set, model, base_model, validation_set=None
    ):
        pass

    @abstractmethod
    def output(self, result):
        return result
