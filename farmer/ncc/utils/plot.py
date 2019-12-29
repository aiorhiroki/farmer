import matplotlib.pyplot as plt
import os
import numpy as np


# 図の保持
class MatPlotManager:
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._figures = {}

    def add_figure(self, title, xy_labels, labels, filename=None):
        assert not(title in self._figures.keys()), "This title already exists."
        self._figures[title] = MatPlot(
            title, xy_labels, labels, self._root_dir, filename=filename)
        return self._figures[title]

    def get_figure(self, title):
        return self._figures[title]


# 学習履歴のプロット
class MatPlot:
    EXTENSION = ".png"

    def __init__(self, title, xy_labels, labels, root_dir, filename=None):
        assert len(labels) > 0 and len(xy_labels) == 2
        if filename is None:
            self._filename = title
        else:
            self._filename = filename
        self._title = title
        self._x_label, self._y_label = xy_labels[0], xy_labels[1]
        self._labels = labels
        self._root_dir = root_dir
        self._series = np.zeros((len(labels), 0))

    def add(self, series, is_update=False):
        series = np.asarray(series).reshape((len(series), 1))
        assert series.shape[0] == self._series.shape[0], 'not same length'
        self._series = np.concatenate([self._series, series], axis=1)
        if is_update:
            self.save()

    def save(self):
        plt.cla()
        for s, l in zip(self._series, self._labels):
            plt.plot(s, label=l)
        plt.legend()
        plt.grid()
        plt.xlabel(self._x_label)
        plt.ylabel(self._y_label)
        plt.title(self._title)
        plt.savefig(os.path.join(self._root_dir,
                                 self._filename+self.EXTENSION))
