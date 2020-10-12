from __future__ import print_function
from parse import parse
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from keras.utils.io_utils import HDF5Matrix
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from core import create_RobusterNet, create_linear_BusterNet
import warnings
warnings.filterwarnings("ignore")


def get_target_idx( xn, ynames ) :
    fmt = '{}_F_{}'
    try :
        img_id, postproc = parse( fmt, xn )
    except :
        img_id = xn.rsplit('_')[0]
        postproc = 'BASE'
    idx = ynames.index( img_id )
    return idx, img_id, postproc


def evaluate_CoMoFoD_performance( Z, XN, Y, YN ) :
    ynames = []
    for yn in YN :
        ynames.append(yn)
    prf_list = []
    correct = []

    plut = {'mapping':{}}
    for xidx, (xn, z) in enumerate( zip( XN, Z ) ) :

        idx, img_id, postproc = get_target_idx( xn, ynames )
        y = Y[idx]

        if postproc not in plut :
            plut[postproc] = []
        ref = y[...,2].ravel() == 0
        hyp = z[...,2].ravel() <= 0.5
        precision, recall, fscore, _ = precision_recall_fscore_support( ref, hyp,
                                                                        pos_label=1,
                                                                        average='binary')

        prf_list.append([precision, recall, fscore])

        plut[postproc].append( [precision, recall, fscore] )

        if postproc == 'BASE' :
            plut['mapping'][xidx] = [idx, fscore]

    print( "INFO: BusterNet Performance on CoMoFoD-CMFD Dataset using the number of correct detections" )
    print("-" * 100)
    for key, res in sorted( plut.items() ) :
        if key == 'mapping' :
            nbase_correct = np.sum(np.row_stack(res)[:, -1] > .5)
            print("{:>4s}: {:>3}".format(key, nbase_correct))
            continue

        # a sample is correct if its F1 score is above 0.5
        nb_correct = np.sum( np.row_stack(res)[:,-1] > .5 )
        print ("{:>4s}: {:>3}".format( key, nb_correct ) )

        # print mean precision, recall, fscore over the postprocessed images
        print(np.mean(np.row_stack(res)[:, 0]))
        print(np.mean(np.row_stack(res)[:, 1]))
        print(np.mean(np.row_stack(res)[:,-1]))


    prf = np.row_stack(prf_list)

    print("INFO: BusterNet Performance on CoMoFoD Dataset using Pixel-Level Evaluation Protocol-B")
    print("-" * 100)
    for name, mu in zip(['Precision', 'Recll', 'F1'], prf.mean(axis=0)):
        print("INFO: {:>9s} = {:.3f}".format(name, mu))

    return plut, prf




def main():
    CoMoFoD_hd5 = 'data/CoMoFoD-CMFD/CoMoFoD-CMFD.hd5'

    X = HDF5Matrix(CoMoFoD_hd5, 'X')
    XN = HDF5Matrix(CoMoFoD_hd5, 'XN')
    Y = HDF5Matrix(CoMoFoD_hd5, 'Y')
    YN = HDF5Matrix(CoMoFoD_hd5, 'YN')

    busternet = create_linear_BusterNet('models/pretrained_busterNet.hd5')

    for layer in busternet.layers:
        layer.trainable = False

    model = create_RobusterNet(busternet, weight_file='models/robusternet_weights.hdf5')

    Z = model.predict(X, verbose=1, batch_size=1)
    plut, prf = evaluate_CoMoFoD_performance(Z, XN, Y, YN)

    llut = {'Bright Change(BC)': range(1, 4), 'Contrast Adjustment(CA)': range(4, 7),
            'Color Reduction(CR)': range(7, 10), 'Image Blurring(IB)': range(10, 13),
            'JPEG Compression(JC)': range(13, 22), 'Noise Adding(NA)': range(22, 25)}

    pyplot.figure(figsize=(12, 8))
    ii = 1
    for key, vals in llut.items():
        ys = []
        xnames = []
        for idx, val in enumerate(vals):
            _, prefix = parse('{}({})', key)
            tkey = prefix + str(idx + 1)
            ys.append(np.mean(np.row_stack(plut[tkey]), axis=0))
            xnames.append(tkey)

        pyplot.subplot(2, 3, ii)
        pyplot.plot(np.array(ys))
        pyplot.xticks(range(len(vals)), xnames, fontsize=12)
        pyplot.legend(['Precision', 'Recall', 'F1 Score'], fontsize=12)
        pyplot.ylim([0, .7])
        ii += 1
    pyplot.savefig('prf_RobusterNet.png')


if __name__ == '__main__':
    main()