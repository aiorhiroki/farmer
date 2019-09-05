import os
import ncc
import keras


class TrainTask:

    def __init__(self, config):
        self.config = config

    def command(
        self,
        model,
        base_model,
        train_set,
        validation_set
    ):

        train_gen, validation_gen = self._do_generate_batch_task(
            train_set, validation_set
        )
        callbacks = self._do_set_callbacks_task(
            base_model, train_set, validation_set
        )
        trained_model = self._do_model_optimization_task(
            model, train_gen, validation_gen, callbacks
        )
        self._do_save_model_task(trained_model, base_model)

    def _do_generate_batch_task(
        self,
        train_set,
        validation_set
    ):
        sequence_args = dict(
            annotations=train_set,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            task=self.config.task,
            batch_size=self.config.batch_size,
            augmentation=self.config.augmentation
        )
        train_gen = ncc.generators.ImageSequence(**sequence_args)

        sequence_args.update(
            annotations=validation_set,
            augmentation=False
        )
        validation_gen = ncc.generators.ImageSequence(**sequence_args)

        return train_gen, validation_gen

    def _do_set_callbacks_task(
        self,
        base_model,
        train_set,
        validation_set
    ):
        best_model_name = 'best_model.h5'
        model_save_file = os.path.join(
            self.config.model_path,
            best_model_name
        )
        if self.config.multi_gpu:
            checkpoint = ncc.callbacks.MultiGPUCheckpointCallback(
                filepath=model_save_file,
                base_model=base_model,
                save_best_only=True,
            )
        else:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=model_save_file,
                save_best_only=True,
            )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            factor=0.1, patience=3, verbose=1
        )
        plot_history = ncc.callbacks.PlotHistory(self.config.learning_path)
        callbacks = [
            checkpoint,
            reduce_lr,
            plot_history
        ]
        if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            iou_history = ncc.callbacks.IouHistory(
                save_dir=self.config.learning_path,
                validation_files=validation_set,
                class_names=self.config.class_names,
                height=self.config.height,
                width=self.config.width
            )
            train_save_dir = os.path.join(self.config.image_path, 'train')
            val_save_dir = os.path.join(self.config.image_path, 'validation')
            generate_sample_result = ncc.callbacks.GenerateSampleResult(
                train_save_dir=train_save_dir,
                val_save_dir=val_save_dir,
                train_files=train_set,
                validation_files=validation_set,
                nb_classes=self.config.nb_classes,
                height=self.config.height,
                width=self.config.width
            )
            callbacks.extend([iou_history, generate_sample_result])
        if self.config.slack_channel and self.config.slack_token:
            if self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
                file_name = os.path.join(
                    self.config.learning_path, 'IoU.png'
                )
            else:
                file_name = os.path.join(
                    self.config.learning_path, 'Metric.png'
                )

            slack_logging = ncc.callbacks.SlackLogger(
                file_name=file_name,
                token=self.config.slack_token,
                channel=self.config.slack_channel,
                title=self.config.model_name
            )
            callbacks.append(slack_logging)
        return callbacks

    def _do_model_optimization_task(
        self,
        model,
        train_gen,
        validation_gen,
        callbacks
    ):

        model.fit_generator(
            train_gen,
            steps_per_epoch=len(train_gen),
            callbacks=callbacks,
            epochs=self.config.epochs,
            validation_data=validation_gen,
            validation_steps=len(validation_gen),
            workers=16 if self.config.multi_gpu else 1,
            max_queue_size=32 if self.config.multi_gpu else 10,
            use_multiprocessing=self.config.multi_gpu
        )

        return model

    def _do_save_model_task(
        self,
        model,
        base_model
    ):
        model_path = os.path.join(self.config.model_dir, 'last_model.h5')
        if self.config.multi_gpu:
            base_model.save(model_path)
        else:
            model.save(model_path)
