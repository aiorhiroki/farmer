from tensorflow.keras import backend as k
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(model, images, labels, layer_id=-2):
    # layer_id = -2 , which means just before final output
    get_fc_layer_output = k.function([model.layers[0].input, k.learning_phase()],
                                     [model.layers[layer_id].output])

    # output in test mode = 0
    features = get_fc_layer_output([images, 0])[0]

    pca = PCA(n_components=2)
    pca.fit(features)

    # Convert the data set to the main component based on the analysis result
    transformed = pca.fit_transform(features)

    # Plot the main componet
    plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels)
    plt.colorbar()
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')

    plt.show()
