import os
import numpy as np
from farmer import ncc
from tensorflow.python import keras

import torch


class TrainTask:
    def __init__(self, config):
        self.config = config

    def command(
            self, model, base_model, train_set, validation_set, trial):

        train_gen, validation_gen = self._do_generate_batch_task(
            train_set, validation_set, trial
        )
        callbacks = self._do_set_callbacks_task(
            base_model, train_set, validation_set
        )
        trained_model = self._do_model_optimization_task(
            model, train_gen, validation_gen, callbacks
        )
        save_model = self._do_save_model_task(trained_model, base_model)

        return save_model

    def _do_generate_batch_task(self, train_set, validation_set, trial):
        np.random.shuffle(train_set)
        if self.config.op_batch_size:
            batch_size = int(trial.suggest_discrete_uniform(
                'batch_size', *self.config.batch_size))
        else:
            batch_size = self.config.batch_size

        sequence_args = dict(
            annotations=train_set,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            task=self.config.task,
            batch_size=batch_size,
            augmentation=self.config.augmentation,
            train_colors=self.config.train_colors,
            input_data_type=self.config.input_data_type
        )

        if self.config.framework == 'tensorflow':
            train_gen = ncc.generators.ImageSequence(**sequence_args)

            sequence_args.update(annotations=validation_set, augmentation=[])
            validation_gen = ncc.generators.ImageSequence(**sequence_args)

        elif self.config.framework == 'pytoch':
            train_gen = ncc.generators.ImageDataset(**sequence_args)

            sequence_args.update(annotations=validation_set, augmentation=[])
            validation_gen = ncc.generators.ImageDataset(**sequence_args)

        return train_gen, validation_gen

    def _do_fetch_checkpoint(self, base_model, model_save_file):
        if self.config.framework == 'tensorflow':
            if self.config.multi_gpu:
                return ncc.callbacks.MultiGPUCheckpointCallback(
                    filepath=model_save_file,
                    base_model=base_model,
                    save_best_only=True,
                )
            else:
                return keras.callbacks.ModelCheckpoint(
                    filepath=model_save_file, save_best_only=True
                )

        elif self.config.framework == 'pytorch':
            return None

    def _do_fetch_scheduler(self):
        if self.config.framework == 'tensorflow':
            if self.config.cosine_decay:
                ncc_scheduler = ncc.schedulers.Scheduler(
                    self.config.cosine_lr_max,
                    self.config.cosine_lr_min,
                    self.config.epochs
                )

                return keras.callbacks.LearningRateScheduler(
                    ncc_scheduler.cosine_decay
                )

            else:
                return keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=10,
                    verbose=1,
                )

        elif self.config.framework == 'pytorch':
            return None

    def _do_fetch_plot_history(self):
        if self.config.framework == 'tensorflow':
            return ncc.callbacks.PlotHistory(
                self.config.learning_path,
                ['loss', 'acc', 'iou_score', 'categorical_crossentropy']
            )

        elif self.config.framework == 'pytorch':
            return None

    def _do_fetch_iou_history(self, validation_set):
        if self.config.framework == 'tensorflow':
            return ncc.callbacks.IouHistory(
                save_dir=self.config.learning_path,
                validation_files=validation_set,
                class_names=self.config.class_names,
                height=self.config.height,
                width=self.config.width,
                train_colors=self.config.train_colors
            )

        elif self.config.framework == 'pytorch':
            return None

    def _do_fetch_generate_sample_result(self, val_save_dir, validation_set):
        if self.config.framework == 'tensorflow':
            return ncc.callbacks.GenerateSampleResult(
                val_save_dir=val_save_dir,
                validation_files=validation_set,
                nb_classes=self.config.nb_classes,
                height=self.config.height,
                width=self.config.width,
                train_colors=self.config.train_colors,
                segmentation_val_step=self.config.segmentation_val_step
            )

        elif self.config.framework == 'pytorch':
            return None

    def _do_fetch_slack_logging(self):
        if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            file_name = os.path.join(self.config.learning_path, "IoU.png")
        else:
            file_name = os.path.join(
                self.config.learning_path, "Metric.png"
            )

        if self.config.framework == 'tensorflow':
            return ncc.callbacks.SlackLogger(
                logger_file=file_name,
                token=self.config.slack_token,
                channel=self.config.slack_channel,
                title=self.config.model_name,
            )

        elif self.config.framework == 'pytorch':
            return None

    def _do_set_callbacks_task(self, base_model, train_set, validation_set):
        best_model_name = "best_model.h5"
        model_save_file = os.path.join(self.config.model_path, best_model_name)
        if self.config.multi_gpu:
            checkpoint = ncc.callbacks.MultiGPUCheckpointCallback(
                filepath=model_save_file,
                base_model=base_model,
                save_best_only=True,
            )
        else:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=model_save_file, save_best_only=True
            )
        if self.config.cosine_decay:
            ncc_scheduler = ncc.schedulers.Scheduler(
                self.config.cosine_lr_max,
                self.config.cosine_lr_min,
                self.config.epochs
            )
            scheduler = keras.callbacks.LearningRateScheduler(
                ncc_scheduler.cosine_decay)
        else:
            scheduler = keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, verbose=1)

        plot_history = ncc.callbacks.PlotHistory(
            self.config.learning_path,
            ['loss', 'acc', 'iou_score', 'categorical_crossentropy']
        )
        callbacks = [checkpoint, scheduler, plot_history]
        if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            iou_history = ncc.callbacks.IouHistory(
                save_dir=self.config.learning_path,
                validation_files=validation_set,
                class_names=self.config.class_names,
                height=self.config.height,
                width=self.config.width,
                train_colors=self.config.train_colors
            )
            val_save_dir = os.path.join(self.config.image_path, "validation")
            generate_sample_result = ncc.callbacks.GenerateSampleResult(
                val_save_dir=val_save_dir,
                validation_files=validation_set,
                nb_classes=self.config.nb_classes,
                height=self.config.height,
                width=self.config.width,
                train_colors=self.config.train_colors,
                segmentation_val_step=self.config.segmentation_val_step
            )
            callbacks.extend([iou_history, generate_sample_result])
        if self.config.slack_channel and self.config.slack_token:
            if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
                file_name = os.path.join(self.config.learning_path, "IoU.png")
            else:
                file_name = os.path.join(
                    self.config.learning_path, "Metric.png"
                )

            slack_logging = ncc.callbacks.SlackLogger(
                logger_file=file_name,
                token=self.config.slack_token,
                channel=self.config.slack_channel,
                title=self.config.model_name,
            )
            callbacks.append(slack_logging)
        return callbacks

    def _do_model_optimization_task(
        self, model, train_gen, validation_gen, callbacks
    ):
        if self.config.framework == 'tensorflow':
            model.fit_generator(
                train_gen,
                steps_per_epoch=len(train_gen),
                callbacks=callbacks,
                epochs=self.config.epochs,
                validation_data=validation_gen,
                validation_steps=len(validation_gen),
                workers=16 if self.config.multi_gpu else 1,
                max_queue_size=32 if self.config.multi_gpu else 10,
                use_multiprocessing=self.config.multi_gpu,
            )

        elif self.config.framework == 'pytorch':
            print(
                '[train_task.py, _do_model_optimization_task, pytorch] under construction'
            )

        return model

    def _do_save_model_task(self, model, base_model):
        model_path = os.path.join(self.config.model_path, "last_model.h5")
        if self.config.multi_gpu:
            base_model.save(model_path)
            return base_model
        else:
            model.save(model_path)
            return model
