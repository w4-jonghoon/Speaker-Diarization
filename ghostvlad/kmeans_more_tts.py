from absl import app
from absl import flags
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

FLAGS = flags.FLAGS
flags.DEFINE_string('embeddings', './embeddings_1000.npz.backup',
                    'Numpy file contains embeddings and targets')

def _get_embeddings(data_filepath):
    embedding_load = np.load(data_filepath)
    labels = np.arange(8).repeat(1000)
    return embedding_load['feats'], labels


def main(args):
    del args # Unused
    embeddings, targets = _get_embeddings(FLAGS.embeddings)
    print(embeddings.shape, targets.shape)

    model = KMeans(init='k-means++', n_clusters=8, random_state=0)
    model.fit(embeddings)
    y_pred = model.labels_

    print(confusion_matrix(targets, y_pred))

if __name__ == '__main__':
    app.run(main)