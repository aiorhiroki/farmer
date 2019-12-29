# coding; utf-8
from keras.utils import plot_model
import os


# layersリストの中にさらにリストがあっても、ネットワークがつながる。
def inst_layers(layers, in_layer):
    x = in_layer
    for layer in layers:
        if isinstance(layer, list):
            x = inst_layers(layer, x)
        else:
            x = layer(x)

    return x


def summary_and_png(model, summary=True, to_png=False, png_file=None):
    if summary:
        model.summary()
    if to_png:
        os.makedirs('summary', exist_ok=True)
        plot_model(model, to_file='summary/'+png_file, show_shapes=True)
