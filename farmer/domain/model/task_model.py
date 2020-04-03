from enum import Enum, IntEnum


class Task(IntEnum):
    CLASSIFICATION = 1
    OBJECT_DETECTION = 2
    SEMANTIC_SEGMENTATION = 3


class Framework(Enum):
    TENSORFLOW = 1
    PYTORCH = 2
