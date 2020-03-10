import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorflow import keras
import torch
from farmer import ncc

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

from segmentation_models_pytorch.utils.losses import (
    DiceLoss, JaccardLoss, CrossEntropyLoss
)

from segmentation_models_pytorch.utils.metrics import (
    IoU, Accuracy
)


class TrainTask:
    def __init__(self, config):
        self.config = config

    def command(
            self, model, base_model, train_set, validation_set, trial, optimizer):

        train_gen, validation_gen = self._do_generate_batch_task(
            train_set, validation_set, trial
        )

        callbacks = self._do_set_callbacks_task(
            base_model, train_set, validation_set
        )

        trained_model = self._do_model_optimization_task(
            model, train_gen, validation_gen, callbacks, optimizer
        )

        saved_model = self._do_save_model_task(trained_model, base_model)

        return saved_model

    def _do_generate_batch_task(self, train_set, validation_set, trial) -> (list, list):
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

            return train_gen, validation_gen

        elif self.config.framework == 'pytorch':
            # ImageDatasetの引数にtransform=DataTransform()を追加する
            train_dataset = ncc.generators.ImageDataset(**sequence_args)

            sequence_args.update(annotations=validation_set, augmentation=[])
            validation_dataset = ncc.generators.ImageDataset(**sequence_args)

            return train_dataset, validation_dataset

        # unexpected case
        return [None], [None]

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
            # TODO: Callbacks of ModelCheckpoint at PyTorch
            return None

    def _do_fetch_scheduler(self, optimizer):
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
            if self.config.cosine_decay:
                ncc_scheduler = ncc.schedulers.Scheduler(
                    self.config.cosine_lr_max,
                    self.config.cosine_lr_min,
                    self.config.epochs
                )

                return torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=ncc_scheduler.cosine_decay
                )

            else:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.5,
                    patience=10,
                    verbose=1
                )

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
            # TODO: 推論の結果画像生成
            return [None]

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
            # TODO: Slackにログを飛ばす
            return None

    def _do_set_callbacks_task(self, base_model, train_set, validation_set):

        if self.config.framework == 'pytorch':
            best_model_name = "best_model.pth"
            return None

        if self.config.framework == 'tensorflow':
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
        self, model, train_gen, validation_gen, callbacks, optimizer
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
            if self.config.multi_gpu:
                train_dataloader = torch.utils.data.DataLoader(
                    train_gen,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=16,
                    worker_init_fn=self.worker_init_fn,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    validation_gen,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=16,
                    worker_init_fn=self.worker_init_fn,
                )

            else:
                train_loader = torch.utils.data.DataLoader(
                    train_gen,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

                valid_loader = torch.utils.data.DataLoader(
                    validation_gen,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

            dataloaders = {
                'train': train_loader,
                'val': valid_loader,
            }

            criterion = self._do_fetch_criterion(self.config.loss)

            model = self._do_train_pytorch_model_task(
                dataloaders,
                self.config.epochs,
                optimizer,
                model,
                criterion,
            )

            """
            # Segmentation_models_pytorchのtrainerを使った場合
            # 参考 https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
            # CrossEntropyは使えない
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            metrics = [IoU(), Accuracy()]

            train_epoch = TrainEpoch(
                model, 
                loss=criterion, 
                metrics=metrics, 
                optimizer=optimizer,
                device=device,
                verbose=True
            )

            valid_epoch = ValidEpoch(
                model, 
                loss=criterion, 
                metrics=metrics, 
                device=device,
                verbose=True
            )

            max_score = 0
            for i in range(self.config.epochs):
                
                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)
                
                # do something (save model, change lr, etc.)
                if max_score < valid_logs['iou_score']:
                    max_score = valid_logs['iou_score']
                    torch.save(model, './best_model.pth')
                    print('Model saved!')
            """

        return model

    def _do_fetch_criterion(self, criterion_name: str):
        if criterion_name == 'dice_loss':
            return DiceLoss()

        elif criterion_name == 'jaccard_loss':
            return JaccardLoss()

        elif criterion_name == 'crossentropy_loss':
            return CrossEntropyLoss()

        else:
            return None

    def worker_init_fn(worker_id, seed=1):
        random.seed(worker_id + seed)
        np.random.seed(worker_id + seed)

    def _do_train_pytorch_model_task(
        self,
        dataloaders,
        epochs,
        optimizer,
        model,
        criterion
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        scheduler = self._do_fetch_scheduler(optimizer)
        ce_loss = self._do_fetch_criterion('crossentropy_loss')
        logs = list()

        for epoch in range(epochs):
            loss_logs = dict(
                train=0.0,
                val=0.0,
                train_ce=0.0,
                val_ce=0.0
            )

            metrics_logs = dict(
                train_iou=list(),
                val_iou=list(),
                train_acc=list(),
                val_acc=list()
            )

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()

                    if self.config.cosine_decay:
                        scheduler.step()

                    optimizer.zero_grad()

                else:
                    model.eval()

                for images, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                    if images.size(0) == 1:
                        continue

                    images = images.to(device)
                    labels = labels.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)

                        loss_ce = ce_loss(outputs, torch.argmax(labels, axis=1).long())  # crossentropyはonehot化しない
                        loss = criterion(outputs, labels.long()) if self.config.loss != "crossentropy_loss" else loss_ce
                        iou = IoU()(outputs, labels.long())
                        acc = Accuracy()(outputs, labels.long())

                        if phase == 'train':
                            loss.backward()

                            loss_logs["train"] += loss.item()
                            loss_logs["train_ce"] += loss_ce.item()
                            metrics_logs["train_iou"].append(iou.item())
                            metrics_logs["train_acc"].append(acc.item())

                            optimizer.step()
                            optimizer.zero_grad()

                        else:
                            _, predict_result = torch.max(outputs.data, 1)

                            loss_logs["val"] += loss.item()
                            loss_logs["val_ce"] += loss_ce.item()
                            metrics_logs["val_iou"].append(iou.item())
                            metrics_logs["val_acc"].append(acc.item())

            print(
                f"Epoch: {epoch+1:03d} - Train loss: {loss_logs['train']} - Val loss: {loss_logs['val']}")

            logs.append(
                {
                    'epoch': epoch + 1,
                    'train_loss': loss_logs["train"],
                    'train_crossentropy': loss_logs["train_ce"],
                    'val_loss': loss_logs["val"],
                    'val_crossentropy': loss_logs["val_ce"],
                    'iou_train': self._do_calculate_avg(metrics_logs["train_iou"]),
                    'acc_train': self._do_calculate_avg(metrics_logs["train_acc"]),
                    'iou_val': self._do_calculate_avg(metrics_logs["val_iou"]),
                    'acc_val': self._do_calculate_avg(metrics_logs["val_acc"]),
                }
            )

        df = pd.DataFrame(logs)
        df.to_csv('log_output.csv', index=False)

        '''
        TODO: callbacks
        1-1. 学習済モデルの保存 (複数GPU or GPU) - on_epoch_end: すべてのエポックの終了時に呼ばれます．
        2-1. スケジューラの取得
        3-1. plot_historyの取得               - 訓練の開始時、全てのエポックの終了時
         plot_history = ncc.callbacks.PlotHistory(
            self.config.learning_path,
            ['loss', 'acc', 'iou_score', 'categorical_crossentropy']
        )
        4-1. iou_historyの取得                - 訓練の開始時、全てのエポックの終了時
        4-2. generate_sample_resultの取得     - 全てのエポックの終了時
        5. Slackへの画像送信 -> 一旦、廃止
        '''

        return model

    def _do_calculate_avg(self, target_list):
        if not target_list:
            return -1

        return round(sum(target_list) / len(target_list), 3)

    def _do_save_model_task(self, model, base_model):
        if self.config.framework == "tensorflow":
            model_path = os.path.join(self.config.model_path, "last_model.h5")

            if self.config.multi_gpu:
                base_model.save(model_path)
                return base_model

            else:
                model.save(model_path)
                return model

        elif self.config.framework == "pytorch":
            model_path = os.path.join(self.config.model_path, "last_model.pth")

            if self.config.multi_gpu:
                torch.save(base_model.state_dict(), model_path)
                return base_model

            else:
                torch.save(model.state_dict(), model_path)
                return model
