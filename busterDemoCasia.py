from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
from core import create_RobusterNet, create_linear_BusterNet
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import precision_recall_fscore_support
np.set_printoptions(threshold=np.inf)
import warnings
warnings.filterwarnings("ignore")

def evaluate_protocol_B(Y, Z) :
    prf_list = []

    for rr, hh in zip( Y, Z ) :
        ref = rr[...,-1].ravel() == 0
        hyp = hh[...,-1].ravel() <= 0.5
        precision, recall, fscore, _ = precision_recall_fscore_support( ref, hyp,
                                                                        pos_label=1,
                                                                        average='binary')
        prf_list.append([precision, recall, fscore])

    prf = np.row_stack( prf_list )
    print( "INFO: BusterNet Performance on CASIA-CMFD Dataset using Pixel-Level Evaluation Protocol-B" )
    print("-" * 100)
    for name, mu in zip( ['Precision', 'Recll', 'F1'], prf.mean(axis=0)):
        print( "INFO: {:>9s} = {:.3f}".format(name, mu))
    return prf


def main():
    print("INFO: this notebook has been tested under keras.version=2.2.2, tensorflow.version=1.8.0")
    print("INFO: here is the info your local")
    print("      keras.version={}".format(keras.__version__))
    print("      tensorflow.version={}".format(tf.__version__))
    print("INFO: consider to the suggested versions if you can't run the following code properly.")

    busternet = create_linear_BusterNet(weight_file='models/pretrained_busterNet.hd5')

    for layer in busternet.layers:
        layer.trainable = False

    model = create_RobusterNet(busternet, weight_file='models/robusternet_weights.hdf5')

    print(model.summary())

    X = HDF5Matrix('data/CASIA-CMFD/CASIA-CMFD-Pos.hd5', 'X')
    Y = HDF5Matrix('data/CASIA-CMFD/CASIA-CMFD-Pos.hd5', 'Y')
    Z = model.predict(X, verbose=1, batch_size=1)

    evaluate_protocol_B(Y, Z)


if __name__ == '__main__':
    main()