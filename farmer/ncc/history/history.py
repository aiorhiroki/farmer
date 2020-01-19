"""Keras, History functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_history(history, result_file='history.csv'):
    """Save history instance as file.

    # Arguments
        history: Returns of fit method.
        result_file: Path to save as csv file. End with '.csv'.

    # Returns
        Save as csv file.
    """
    df = pd.DataFrame(history.history)
    df.to_csv(result_file, sep=',', index_label='epoch')


def get_array(files):
    """Convert file to numpy array.

    # Arguments
        files: Path to file, saved by above save_history method.

    # Returns
        labels: Dictionary, Keys(file_path) and Values(metrics name).
        values: Dictionary, Keys(file_path) and Values(metrics value).
    """
    labels, values = {}, {}
    for file in files:
        df = pd.read_csv(file)
        labels[file], values[file] = df.columns.tolist(), df.values
    
    return labels, values


def show_history(metrics='acc', average=False, *files):
    """Show history.

    # Arguments
        metrics: Metrics name. If 'acc', you can see 'acc' and 'val_acc'.
        average: Moving average. (e.g. 3 and 5)
        files: Path to file, saved by above save_history method. It receives multiple files.

    # Returns
        Show as integrated graph.
    """

    labels, values = get_array(files)
    colors = ["b", "g", "r", "c", "m", "y", "b", "w"]
    plt.figure(figsize=(12, 8))
    for i, key in enumerate(values.keys()):
        if average:
            for column in range(1, values[key].shape[1]):
                values[key][:, column] = np.convolve(values[key][:, column], np.ones(average)/float(average), 'same')
                values[key] = values[key][average//2:-((average//2)+1)]
                
        plt.plot(values[key][:, 0], values[key][:, labels[key].index(metrics)],
                 colors[i],
                 alpha=0.3,
                 label=key[:-4]+' '+metrics)

        if 'val_'+metrics in labels[key]:
            plt.plot(values[key][:, 0], values[key][:, labels[key].index('val_'+metrics)],
                     colors[i],
                     alpha=0.9,
                     label=key[:-4]+' '+'val_'+metrics)

    plt.title('History')
    plt.xlabel('Epochs')
    plt.ylabel(metrics)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.00), fontsize=12)
    plt.grid(color='gray', alpha=0.3)
    plt.show()
