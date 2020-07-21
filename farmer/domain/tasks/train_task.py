import os
import numpy as np
from farmer import ncc
from tensorflow.python import keras
from optuna.integration import KerasPruningCallback


class TrainTask:
    def __init__(self, config):
        self.config = config

    def command(
            self, model, base_model, train_set, validation_set, trial):

        train_gen, validation_gen = self._do_generate_batch_task(
            train_set, validation_set, trial
        )
        callbacks = self._do_set_callbacks_task(
            base_model, train_set, validation_set, trial
        )
        trained_model = self._do_model_optimization_task(
            model, train_gen, validation_gen, callbacks, trial
        )
        save_model = self._do_save_model_task(trained_model, base_model, trial)

        return save_model

    def _do_generate_batch_task(self, train_set, validation_set, trial):
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
            augmentation=self.config.augmentation,
            train_colors=self.config.train_colors,
            input_data_type=self.config.input_data_type
        )
        train_dataset = ncc.generators.Dataset(**sequence_args)
        sequence_args.update(annotations=validation_set, augmentation=[])
        validation_dataset = ncc.generators.Dataset(**sequence_args)

        train_dataloader = ncc.generators.Dataloder(
            train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = ncc.generators.Dataloder(
            validation_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, validation_dataloader

    def _do_set_callbacks_task(
            self, base_model, train_set, validation_set, trial):

        best_model_name = "best_model.h5"
        if trial:
            # result_dir/trial#/model/
            trial_model_path = self.config.model_path.split('/')
            trial_model_path.insert(-1, f"trial{trial.number}")
            if trial_model_path[0] == '':
                trial_model_path[0] = '/'
            model_save_file = os.path.join(*trial_model_path, best_model_name)
        else:
            model_save_file = os.path.join(
                self.config.model_path, best_model_name)
        # Save Model Checkpoint
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
        # Learning Rate Schedule
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
        # Plot History
        if trial:
            # result_dir/trial#/learning/
            trial_learning_path = self.config.learning_path.split('/')
            trial_learning_path.insert(-1, f"trial{trial.number}")
            if trial_learning_path[0] == '':
                trial_learning_path[0] = '/'
            learning_path = os.path.join(*trial_learning_path)
        else:
            learning_path = self.config.learning_path

        plot_history = ncc.callbacks.PlotHistory(
            learning_path,
            ['loss', 'acc', 'iou_score', 'categorical_crossentropy']
        )

        callbacks = [checkpoint, scheduler, plot_history]

        if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            # Plot IoU History
            iou_history = ncc.callbacks.IouHistory(
                save_dir=learning_path,
                validation_files=validation_set,
                class_names=self.config.class_names,
                height=self.config.height,
                width=self.config.width,
                train_colors=self.config.train_colors
            )

            # Predict validation
            if trial:
                # result_dir/trial#/image/validation/
                trial_image_path = self.config.image_path.split('/')
                trial_image_path.insert(-1, f"trial{trial.number}")
                if trial_image_path[0] == '':
                    trial_image_path[0] = '/'
                val_save_dir = os.path.join(*trial_image_path, "validation")
            else:
                val_save_dir = os.path.join(
                    self.config.image_path, "validation")
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

            if self.config.optuna:
                # Trial prune for Optuna
                callbacks.append(KerasPruningCallback(trial, 'dice'))

        elif self.config.task == ncc.tasks.Task.CLASSIFICATION:
            if self.config.input_data_type == "video":
                if trial:
                    batch_model_path = os.path.join(
                        trial_model_path, "batch_model.h5")
                else:
                    batch_model_path = os.path.join(
                        self.config.model_path, "batch_model.h5")

                batch_checkpoint = ncc.callbacks.BatchCheckpoint(
                    learning_path,
                    batch_model_path,
                    token=self.config.slack_token,
                    channel=self.config.slack_channel,
                    period=self.config.batch_period
                )
                callbacks.append(batch_checkpoint)

            if self.config.optuna:
                # Trial prune for Optuna
                callbacks.append(KerasPruningCallback(trial, 'val_acc'))

        if self.config.slack_channel and self.config.slack_token:
            if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
                file_name = os.path.join(learning_path, "IoU.png")
            else:
                file_name = os.path.join(learning_path, "acc.png")

            slack_logging = ncc.callbacks.SlackLogger(
                logger_file=file_name,
                token=self.config.slack_token,
                channel=self.config.slack_channel,
                title=self.config.model_name,
            )
            callbacks.append(slack_logging)
        return callbacks

    def _do_model_optimization_task(
        self, model, train_gen, validation_gen, callbacks, trial
    ):

        try:
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
        except KeyboardInterrupt:
            import sys
            # When stoping by Ctrl-C, save model as last_model.h5
            save_model = self._do_save_model_task(model, None, trial)
            print("\nstop training. save last model:", save_model)
            sys.exit()

        return model

    def _do_save_model_task(self, model, base_model, trial):
        last_model_name = "last_model.h5"
        if trial:
            # result_dir/trial#/model/
            trial_model_path = self.config.model_path.split('/')
            trial_model_path.insert(-1, f"trial{trial.number}")
            if trial_model_path[0] == '':
                trial_model_path[0] = '/'
            model_path = os.path.join(*trial_model_path, last_model_name)
        else:
            model_path = os.path.join(self.config.model_path, last_model_name)
        # Last model save
        if self.config.multi_gpu:
            base_model.save(model_path)
            return base_model
        else:
            model.save(model_path)
            return model
