import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from farmer import ncc
from farmer.ncc import schedulers
from optuna.integration import TFKerasPruningCallback


class TrainTask:
    def __init__(self, config):
        self.config = config

    def command(
            self, model, training_set, validation_set, trial):

        train_dataset, valid_dataset = self._do_generate_batch_task(
            training_set, validation_set
        )
        callbacks = self._do_set_callbacks_task(
            model, train_dataset, valid_dataset, trial
        )
        trained_model = self._do_model_optimization_task(
            model, train_dataset, valid_dataset, callbacks
        )
        save_model = self._do_save_model_task(trained_model)

        return save_model

    def _do_generate_batch_task(self, training_set, validation_set):
        sequence_args = dict(
            annotations=training_set,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            mean=self.config.mean,
            std=self.config.std,
            augmentation=self.config.train_params.augmentation,
            augmix=self.config.train_params.augmix,
            train_colors=self.config.train_colors,
            input_data_type=self.config.input_data_type
        )

        if self.config.task == ncc.tasks.Task.CLASSIFICATION:
            train_set = ncc.generators.ClassificationDataset(**sequence_args)

            sequence_args.update(
                annotations=validation_set,
                mean=np.zeros(3),
                std=np.ones(3),
                augmentation=[]
            )
            valid_set = ncc.generators.ClassificationDataset(**sequence_args)

        elif self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            train_set = ncc.generators.SegmentationDataset(**sequence_args)

            sequence_args.update(
                annotations=validation_set,
                mean=np.zeros(3),
                std=np.ones(3),
                augmentation=[]
            )
            valid_set = ncc.generators.SegmentationDataset(**sequence_args)

        return train_set, valid_set

    def _do_set_callbacks_task(
            self, base_model, train_dataset, valid_dataset, trial):

        # Save Model Checkpoint
        # result_dir/model/
        model_save_file = os.path.join(self.config.model_path, "best_model.h5")
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_save_file, save_best_only=True
        )

        # Learning Rate Schedule
        if self.config.train_params.scheduler:
            funcs = self.config.train_params.scheduler["functions"]
            scheduler_name = next(iter(funcs.keys()))
            scheduler_params = next(iter(funcs.values()))
            scheduler_params.update(
                dict(
                    n_epoch=self.config.epochs,
                    base_lr=self.config.train_params.learning_rate
                )
            )
            scheduler = keras.callbacks.LearningRateScheduler(
                getattr(schedulers, scheduler_name)(**scheduler_params))
        else:
            scheduler = keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, verbose=1)

        # Plot History
        # result_dir/learning/
        learning_path = self.config.learning_path

        plot_history = ncc.callbacks.PlotHistory(
            learning_path,
            ['loss', 'acc', 'iou_score', 'f1-score']
        )

        plot_learning_rate = ncc.callbacks.PlotLearningRate(learning_path)

        callbacks = [checkpoint, scheduler, plot_history, plot_learning_rate]

        if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            # Plot IoU History
            iou_history = ncc.callbacks.IouHistory(
                save_dir=learning_path,
                valid_dataset=valid_dataset,
                class_names=self.config.class_names,
                batch_size=self.config.train_params.batch_size
            )

            # Predict validation
            # result_dir/image/validation/
            val_save_dir = os.path.join(self.config.image_path, "validation")

            generate_sample_result = ncc.callbacks.GenerateSampleResult(
                val_save_dir=val_save_dir,
                valid_dataset=valid_dataset,
                nb_classes=self.config.nb_classes,
                batch_size=self.config.train_params.batch_size,
                segmentation_val_step=self.config.segmentation_val_step
            )
            callbacks.extend([iou_history, generate_sample_result])

            if self.config.optuna:
                # Trial prune for Optuna
                callbacks.append(
                    TFKerasPruningCallback(trial, 'val_f1-score'))

        elif self.config.task == ncc.tasks.Task.CLASSIFICATION:
            if self.config.input_data_type == "video":
                # result_dir/model/
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
                callbacks.append(
                    TFKerasPruningCallback(trial, 'val_acc'))

        if self.config.slack_channel and self.config.slack_token:
            if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
                file_name = os.path.join(learning_path, "IoU.png")
            else:
                file_name = os.path.join(learning_path, "acc.png")

            slack_logging = ncc.callbacks.SlackLogger(
                logger_file=file_name,
                token=self.config.slack_token,
                channel=self.config.slack_channel,
                title=self.config.train_params.model_name,
            )
            callbacks.append(slack_logging)

        # Early Stoppoing
        if self.config.early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=self.config.monitor,
                patience=self.config.patience,
                mode='auto'
            )
            callbacks.append(early_stopping)

        return callbacks

    def _do_model_optimization_task(
        self, model, train_dataset, val_dataset, callbacks
    ):
        if self.config.generator:
            train_gen = ncc.generators.Dataloder(
                train_dataset,
                batch_size=self.config.train_params.batch_size,
                shuffle=True
            )
            valid_gen = ncc.generators.Dataloder(
                val_dataset,
                batch_size=self.config.train_params.batch_size,
                shuffle=False
            )
        else:
            val_data = [d for d in val_dataset]
            train_data = [d for d in train_dataset]
            val = [np.stack(samples, axis=0) for samples in zip(*val_data)]
            train = [np.stack(samples, axis=0) for samples in zip(*train_data)]
            valid_ds = tf.data.Dataset.from_tensor_slices((val[0], val[1]))
            train_ds = tf.data.Dataset.from_tensor_slices((train[0], train[1]))

            train_gen = train_ds.shuffle(len(train_dataset)).batch(
                    self.config.train_params.batch_size
            )
            valid_gen = valid_ds.batch(self.config.train_params.batch_size)

        class_weights = self.config.train_params.classification_class_weight

        try:
            model.fit(
                train_gen,
                steps_per_epoch=len(train_gen),
                callbacks=callbacks,
                epochs=self.config.epochs,
                validation_data=valid_gen,
                validation_steps=len(valid_gen),
                workers=16 if self.config.multi_gpu else 1,
                max_queue_size=32 if self.config.multi_gpu else 10,
                use_multiprocessing=self.config.multi_gpu,
                class_weight=class_weights,
            )
        except KeyboardInterrupt:
            import sys
            # When stoping by Ctrl-C, save model as last_model.h5
            save_model = self._do_save_model_task(model, None)
            print("\nstop training. save last model:", save_model)
            sys.exit()

        return model

    def _do_save_model_task(self, model):
        # result_dir/model/
        model_path = os.path.join(self.config.model_path, "last_model.h5")
        model.save(model_path)
        return model
