# coding: utf-8

# Deep Learningの学習精度をプロットする.

import matplotlib.pyplot as plt


def plot_history(history, filename=None):
    # 精度の履歴をプロット
    if 'acc' in history.history:
        plt.plot(history.history['acc'], "o-", label="accuracy")
        if 'val_acc' in history.history:
            plt.plot(history.history['val_acc'], "o-", label="validation_accuracy")
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.ylim([0, 1.05])
        plt.legend(loc="lower right")
        if filename:
            plt.savefig(filename + '_acc.png')
        plt.show()

    # 損失の履歴をプロット
    if 'loss' in history.history:
        plt.plot(history.history['loss'], "o-", label="loss",)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], "o-", label="val_loss")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='lower right')
        if filename:
            plt.savefig(filename + '_loss.png')
        plt.show()
