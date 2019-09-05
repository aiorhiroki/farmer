import os
import shutil
import unittest
from farmer.domain.model.trainer_model import Trainer
from farmer.domain.tasks.eda_task import EdaTask


class EdaTaskTest(unittest.TestCase):
    def setUp(self):
        self.config = Trainer(
            target_dir="target_VOC_path",
            nb_classes=22,
            epochs=100,
            batch_size=2,
            optimizer="adam",
            augmentation="no",
            model_name="deeplab_v3",
            learning_rate=0.01,
            width=512,
            height=256,
            backbone="xception",
            input_dir="image",
            mask_dir="label",
            root_dir="test_result",
            info_path="test_result/20180905_1512/info",
            class_names="0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l",
        )

        self.eda_task = EdaTask(self.config)

        os.makedirs(self.config.info_path)

    def tearDown(self):
        shutil.rmtree(self.config.root_dir)

    def test_do_save_params_task(self):
        # ファイルが作られるかのみ確認。中身は確認していない。
        self.eda_task._do_save_params_task()
        self.assertTrue(
            os.path.exists(
                os.path.join(self.config.info_path, "parameter.txt")
            )
        )

    def test_do_post_config_task(self):
        # milkを起動してテストするのは面倒だから何もしない。
        self.eda_task._do_post_config_task()
