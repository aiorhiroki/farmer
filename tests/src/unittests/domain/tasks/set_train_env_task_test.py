import unittest
from apps.domain.model.trainer_model import Trainer
from apps.domain.tasks.set_train_env_task import SetTrainEnvTask


class SetTrainEnvTaskTest(unittest.TestCase):
    def setUp(self):
        self.config = Trainer(
            target_dir="target_path",
            epochs=100,
            batch_size=2,
            nb_classes=10,
            width=100,
            height=100,
            result_path="test_result/20180905_1512",
        )
        self.set_train_env_task = SetTrainEnvTask(self.config)

    def tearDown(self):
        pass

    def test_do_set_random_seed_task(self):
        self.set_train_env_task._do_set_random_seed_task(seed=1)

    def test_do_set_cpu_gpu_devices_task(self):
        self.set_train_env_task._do_set_cpu_gpu_devices_task(self.config.gpu)

    def test_do_create_dirs_task(self):
        self.set_train_env_task._do_create_dirs_task(self.config.result_path)
