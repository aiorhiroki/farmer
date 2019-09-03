import numpy as np
import ncc


class PredictClassificationTask:

    def __init__(self, config):
        self.config = config

    def command(self, test_set, model):
        prediction_gen = self._do_generate_batch_task(
            test_set
        )
        prediction = self._do_classification_predict(
            model, prediction_gen
        )
        return prediction

    def _do_generate_batch_task(self, annotation_set):
        test_gen = ncc.generators.ImageSequence(
            annotations=self.config.test_files,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            task=self.config.task,
            batch_size=self.config.batch_size
        )
        return test_gen

    def _do_classification_predict(
        self,
        model,
        annotation_gen,
        save_npy=False
    ):
        prediction = model.predict_generator(
            annotation_gen,
            steps=len(annotation_gen),
            workers=16 if self.configmulti_gpu else 1,
            max_queue_size=32 if self.configmulti_gpu else 10,
            use_multiprocessing=self.configmulti_gpu,
            verbose=1
        )
        if save_npy:
            np.save('{}.npy'.format(self.config.model_name), prediction)
        return prediction
